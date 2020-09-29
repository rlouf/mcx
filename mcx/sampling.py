import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from tqdm import tqdm
import warnings

import mcx
from mcx import sample_forward
from mcx.core import compile_to_logpdf


__all__ = ["sampler", "iterative_sampler", "sequential"]


# -------------------------------------------------------------------
#                 == THE BATCH SAMPLING RUNTIME ==
# -------------------------------------------------------------------


class sampler(object):
    """The batch sampling runtime.

    This runtime is encountered in every probabilistic programming library
    (PPL). It allows the user to fetch a pre-defined number of samples from a
    model's posterior distribution.

    While this is undoubtedly the fastest way to obtain samples, it comes at a
    cost in mosts PPLs: to obtain more samples one needs to re-run the inference
    entirely. In MCX the runtime keeps track of the chains' current state so
    that it is possible to get more samples:

    ```python
    trace = sampler.run()  # gives an initial 1,000 samples
    longer_trace = sample.run(5_000)

    final_trace = trace + longer_trace
    ```

    """

    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        model: mcx.model,
        program,
        num_chains: int = 4,
        **observations,
    ):
        """Initialize the batch sampling runtime.

        The runtime is initialized in 4 steps:

        1. Validate the observations that are passed to the model. In
        particular, make sure that all variables defined as input to the
        generative model are provided as well as its output.
        2. Build and compile the loglikelihood from the model's logpdf and the
        observations. This can take some time for large datasets.
        4. Flatten the loglikelihood so the inference algorithms only need to
        deal with flat arrays.
        5. Get initial positions from the program.
        6. Get a function that returns a kernel given its parameters from the
        program.

        Parameters
        ----------
        rng_key
            The key passed to JAX's random number generator. The runtime is
            in charge of splitting the key as it is being used.
        model
            The model whose posterior we want to sample.
        program
            The program that will be used to sampler the posterior.
        num_chains
            The number of chains that will be used concurrently for sampling.
        observations
            The variables we condition on and their values.

        Returns
        -------
        A sampler object.

        """
        print("sampler: build the loglikelihood")
        validate_conditioning_variables(model, **observations)
        loglikelihood = build_loglikelihood(model, **observations)

        print("sampler: find the initial states")
        initial_positions, unravel_fn = get_initial_position(
            rng_key, model, num_chains, **observations
        )
        loglikelihood = flatten_loglikelihood(loglikelihood, unravel_fn)
        initial_state = program.states(initial_positions, loglikelihood)

        print("sampler: build and compile the inference kernel")
        kernel_factory = program.kernel_factory(loglikelihood)
        kernel_factory = jax.jit(kernel_factory, static_argnums=(0, 1, 2))

        self.is_warmed_up = False
        self.rng_key = rng_key
        self.num_chains = num_chains
        self.program = program
        self.kernel_factory = kernel_factory
        self.initial_state = initial_state
        self.state = initial_state
        self.unravel_fn = unravel_fn

    def warmup(self, num_warmup_steps: int = 1000, accelerate: bool = False, **kwargs):
        """Warmup the sampler.

        Warmup is necessary to get values for the program's parameters that are
        adapted to the geometry of the posterior distribution. While run will
        automatically run the warmup if it hasn't been done before, this method
        gives access to the trace for the warmup phase and the values of the
        parameters for diagnostics.

        Parameters
        ----------
        num_warmup_steps
            The number of warmup_steps to perform.
        progress_bar
            If True the progress of the warmup will be displayed. Otherwise it
            will use `lax.scan` to iterate (which is potentially faster).
        kwargs
            Parameters to pass to the program's warmup.

        Returns
        -------
        chain
            The trace of the warmup phase.
        parameters
            The value of parameters that will be used in the sampling phase.

        """
        if self.is_warmed_up:
            warnings.warn(
                "You are trying to warmup a sampler that has already "
                "been warmed up. MCX will re-launch the warmup from the "
                "sampler's initial position. If your use case requires "
                "a different behavior please raise an issue on "
                "https://github.com/rlouf/mcx.",
                UserWarning,
            )
            self.state = self.initial_state

        print(
            f"sampler: warmup {self.num_chains:,} chains for {num_warmup_steps:,} iterations"
        )
        chain_state, parameters = self.program.warmup(
            self.rng_key,
            self.state,
            self.kernel_factory,
            self.num_chains,
            num_warmup_steps,
            accelerate,
            **kwargs,
        )
        self.state = chain_state
        self.parameters = parameters
        self.is_warmed_up = True
        return chain_state, parameters

    def run(
        self,
        num_samples: int = 1000,
        num_warmup_steps: int = 1000,
        accelerate: bool = False,
        **warmup_kwargs,
    ) -> np.DeviceArray:
        """Run the posterior inference.

        For convenience we automatically run the warmup if it hasn't been run
        independently previously. Samples taken during the warmup phase are
        discarded by default. To keep them you can run:

        Parameters
        ----------
        num_samples
            The number of samples to take from the posterior distribution.
        num_warmup_steps
            The number of warmup_steps to perform.
        accelerate
            If False the progress of the warmup and samplig will be displayed.
            Otherwise it will use `lax.scan` to iterate (which is potentially
            faster).
        warmup_kwargs
            Parameters to pass to the program's warmup.

        Returns
        -------
        trace
            A Trace object that contains the chains, some information about
            the inference process (e.g. divergences for programs in the
            HMC family).

        """
        if not self.is_warmed_up:
            self.warmup(num_warmup_steps, accelerate, **warmup_kwargs)

        _, self.rng_key = jax.random.split(self.rng_key)
        rng_keys = jax.random.split(self.rng_key, num_samples)
        state = self.state

        print(f"sampler: draw {num_samples:,} samples from {self.num_chains:,} chains")

        @jax.jit
        def update_chains(rng_key, parameters, chain_state):
            kernel = self.kernel_factory(*parameters)
            new_chain_state, info = kernel(rng_key, chain_state)
            return new_chain_state, info

        # The progress bar is an important indicator for exploratory analysis,
        # while lax.scan is optimal for production environments where speed is
        # needed (an no one is there to look at the progress bar).  Note that
        # for small sample sizes lax.scan is not dramatically faster normal
        # iteration and lax. You thus are not losing much by using it for
        # initial exploration.
        if accelerate:

            @jax.jit
            def update_scan(carry, key):
                state, parameters = carry
                keys = jax.random.split(key, self.num_chains)
                state, info = jax.vmap(update_chains, in_axes=(0, 0, 0))(
                    keys, parameters, state
                )
                return (state, parameters), (state, info)

            last_state, chain = jax.lax.scan(
                update_scan, (state, self.parameters), rng_keys
            )

        else:

            chain = []
            with tqdm(rng_keys, unit="samples") as progress:
                progress.set_description(
                    "Collecting {:,} samples across {:,} chains".format(
                        num_samples, self.num_chains
                    ),
                    refresh=False,
                )
                for key in progress:
                    keys = jax.random.split(key, self.num_chains)
                    state, info = jax.vmap(update_chains, in_axes=(0, 0, 0))(
                        keys, self.parameters, state
                    )
                    chain.append((state, info))

        # chain
        # with progress bar is of format [(state, info) ,(state, info), (state, info)]
        # with scan [all_states, all_infos]
        # I believe second format is easier to unpack later and should be preferred

        self.state = state
        trace = self.program.make_trace(chain, self.unravel_fn)

        return trace


# -------------------------------------------------------------------
#                 == THE ITERATIVE SAMPLING RUNTIME ==
# -------------------------------------------------------------------


def iterative_sampler(
    rng_key, model, program, num_warmup_steps=1000, num_chains=4, **kwargs
):
    """ The generator runtime """

    init, warmup, build_kernel, to_trace, adapt_loglikelihood = program

    validate_conditioning_variables(model, **kwargs)
    loglikelihood = build_loglikelihood(model, **kwargs)

    print("Draw initial states.")
    initial_position, unravel_fn = get_initial_position(
        rng_key, model, num_chains, **kwargs
    )
    loglikelihood = flatten_loglikelihood(loglikelihood, unravel_fn)
    loglikelihood = adapt_loglikelihood(loglikelihood)
    initial_state = jax.vmap(init, in_axes=(0, None))(
        initial_position, jax.value_and_grad(loglikelihood)
    )

    print("Warmup the chains.")
    _, rng_key = jax.random.split(rng_key)
    parameters, state = warmup(rng_key, initial_state, loglikelihood, num_warmup_steps)

    print("Compile the log-likelihood.")
    loglikelihood = jax.jit(loglikelihood)

    print("Build and compile the inference kernel.")
    kernel_builder = build_kernel(loglikelihood)

    @jax.jit
    def update_chains(rng_key, parameters, state):
        kernel = kernel_builder(parameters)
        new_states, info = kernel(rng_key, state)
        return new_states, info

    def run(rng_key, state, parameters):
        while True:
            _, rng_key = jax.random.split(rng_key)
            keys = jax.random.split(rng_key, num_chains)
            state, info = jax.vmap(update_chains, in_axes=(0, 0, 0))(
                keys, parameters, state
            )
            yield (state, info)

    return run(rng_key, initial_state, parameters)


# -------------------------------------------------------------------
#               == THE SEQUENTIAL EXECUTION MODEL ==
# -------------------------------------------------------------------


class sequential(object):
    def __init__(
        self, rng_key, model, program, num_samples=1000, num_warmup_steps=1000
    ):
        """Sequential Markov Chain Monte Carlo sampling."""
        self.model = model
        self.program = program
        self.num_samples = num_samples
        self.num_warmup_steps = num_warmup_steps
        self.rng_key = rng_key

        init, warmup, build_kernel, to_trace, adapt_loglikelihood = self.program
        self.prg_init = init
        self.prg_warmup = warmup
        self.prg_build_kernel = build_kernel
        self.prg_to_trace = to_trace
        self.prg_adapt_loglikelihood = adapt_loglikelihood

        self.state = None

    def _initialize(self, **kwargs):
        loglikelihood = build_loglikelihood(self.model, **kwargs)
        initial_position, self.unravel_fn = get_initial_position(
            self.rng_key, self.model, self.num_samples, **kwargs
        )
        loglikelihood = flatten_loglikelihood(loglikelihood, self.unravel_fn)
        loglikelihood = self.prg_adapt_loglikelihood(loglikelihood)
        initial_state = jax.vmap(self.prg_init, in_axes=(0, None))(
            initial_position, jax.value_and_grad(loglikelihood)
        )
        return initial_state

    def _update_loglikelihood(self, **kwargs):
        loglikelihood = build_loglikelihood(self.model, **kwargs)
        loglikelihood = flatten_loglikelihood(loglikelihood, self.unravel_fn)
        loglikelihood = self.prg_adapt_loglikelihood(loglikelihood)
        loglikelihood = jax.jit(loglikelihood)
        return loglikelihood

    def _update_kernel(self, loglikelihood, parameters):
        kernel = self.prg_build_kernel(loglikelihood, parameters)
        kernel = jax.jit(kernel)
        return kernel

    def update(self, **kwargs):
        _, self.rng_key = jax.random.split(self.rng_key)

        validate_conditioning_variables(self.model, **kwargs)

        if self.state is None:
            self.state = self._initialize(**kwargs)

        # Since the data changes the log-likelihood, and thus the
        # kernel, need to be updated.
        #
        # Although there is no mention of this in the aforementionned
        # papers, we re-run the warmup to adapt the kernel parameters
        # to the new posterior geometry. Unlike the initial warmup, however,
        # we re-start the chains at the initial position.
        loglikelihood = self._update_loglikelihood(**kwargs)
        parameters, _ = self.prg_warmup(
            self.state, loglikelihood, self.num_warmup_steps
        )
        kernel = self._update_kernel(loglikelihood, parameters)

        @jax.jit
        def update_chains(state, rng_key):
            keys = jax.random.split(rng_key, self.num_samples)
            new_states, info = jax.vmap(kernel, in_axes=(0, 0))(keys, state)
            return new_states

        state = self.state

        rng_keys = jax.random.split(self.rng_key, self.num_samples)
        with tqdm(rng_keys, unit="samples") as progress:
            progress.set_description(
                "Collecting {:,} samples".format(self.num_samples),
                refresh=False,
            )
            for key in progress:
                state = update_chains(state, key)
        self.state = state

        trace = self.prg_to_trace(self.state, self.unravel_fn)

        return trace


#
# SHARED UTILITIES
#


def validate_conditioning_variables(model, **kwargs):
    """Check that all variables passed as arguments to the sampler
    are random variables or arguments to the sampler. And converserly
    that all of the model definition's positional arguments are given
    a value.
    """
    conditioning_vars = set(kwargs.keys())
    model_randvars = set(model.random_variables)
    model_args = set(model.arguments)
    available_vars = model_randvars.union(model_args)

    # The variables passed as an argument to the initialization (variables
    # on which the logpdf is conditionned) must be either a random variable
    # or an argument to the model definition.
    if not available_vars.issuperset(conditioning_vars):
        unknown_vars = list(conditioning_vars.difference(available_vars))
        unknown_str = ", ".join(unknown_vars)
        raise AttributeError(
            "You passed a value for {} which are neither random variables nor arguments to the model definition.".format(
                unknown_str
            )
        )

    # The user must provide a value for all of the model definition's
    # positional arguments.
    model_posargs = set(model.posargs)
    if model_posargs.difference(conditioning_vars):
        missing_vars = model_posargs.difference(conditioning_vars)
        missing_str = ", ".join(missing_vars)
        raise AttributeError(
            "You need to specify a value for the following arguments: {}".format(
                missing_str
            )
        )


def build_loglikelihood(model, **kwargs):
    artifact = compile_to_logpdf(model.graph, model.namespace)
    logpdf = artifact.compiled_fn
    loglikelihood = jax.partial(logpdf, **kwargs)
    return loglikelihood


def get_initial_position(rng_key, model, num_chains, **kwargs):
    conditioning_vars = set(kwargs.keys())
    model_randvars = set(model.random_variables)
    to_sample_vars = model_randvars.difference(conditioning_vars)

    samples = sample_forward(rng_key, model, num_samples=num_chains, **kwargs)
    initial_positions = dict((var, samples[var]) for var in to_sample_vars)

    # A naive way to go about flattening the positions is to transform the
    # dictionary of arrays that contain the parameter value to a list of
    # dictionaries, one per position and then unravel the dictionaries.
    # However, this approach takes more time than getting the samples in the
    # first place.
    #
    # Luckily, JAX first sorts dictionaries by keys
    # (https://github.com/google/jax/blob/master/jaxlib/pytree.cc) when
    # raveling pytrees. We can thus ravel and stack parameter values in an
    # array, sorting by key; this gives our flattened positions. We then build
    # a single dictionary that contains the parameters value and use it to get
    # the unraveling function using `unravel_pytree`.
    positions = np.stack(
        [np.ravel(samples[s]) for s in sorted(initial_positions.keys())], axis=1
    )

    # np.atleast_1d is necessary to handle single chains
    sample_position_dict = {
        parameter: np.atleast_1d(values)[0]
        for parameter, values in initial_positions.items()
    }
    _, unravel_fn = ravel_pytree(sample_position_dict)

    return positions, unravel_fn


def flatten_loglikelihood(logpdf, unravel_fn):
    def flattened_logpdf(array):
        kwargs = unravel_fn(array)
        return logpdf(**kwargs)

    return flattened_logpdf
