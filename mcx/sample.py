from datetime import datetime
from typing import Callable, Iterator, Tuple

import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree as jax_ravel_pytree
from tqdm import tqdm

import mcx
from mcx import sample_forward
from mcx.compiler import compile_to_loglikelihoods, compile_to_logpdf
from mcx.jax import ravel_pytree as mcx_ravel_pytree
from mcx.trace import Trace

__all__ = ["sampler"]


# -------------------------------------------------------------------
#                 == THE LINEAR SAMPLING RUNTIME ==
# -------------------------------------------------------------------


class sampler(object):
    """The linear sampling runtime.

    This runtime is encountered in every probabilistic programming library
    (PPL). It allows the user to fetch a pre-defined number of samples from the
    model's posterior distribution. The output is one or several chains that
    contain the successive samples from the posterior.

    While this is undoubtedly the fastest way to obtain samples, it comes at a
    cost in mosts PPLs: to obtain more samples one needs to re-run the inference
    entirely. In MCX the runtime keeps track of the chains' current state so
    that it is possible to get more samples:

        >>> trace = sampler.run()  # gives an initial 1,000 samples
        ... longer_trace = sample.run(5_000)
        ... final_trace = trace + longer_trace

    This runtime comes with two execution model: batch and iterative sampling.
    In the batch model, the one most PPL users are used to, we can fetch a set
    number of samples:

        >>> trace = sampler.run(1_000)

    Executing sampling this way returns a Trace object that contains both the
    samples and additional information about the sampling process. The iterative
    model allows you to fetch samples in a for loop:

        >>> for sample in samples:
        ...     do_something(sample)

    This returns a pair (chain state, sampling info) at each iteration. You can
    take advantage of this execution model for example for dynamic stopping or
    logging samples to be used in, say, tensorboard. No need for any callback
    logic.

    Called with its default values, the `run` method will be as fast as getting
    samples iteratively. We can however make it sensibly faster by calling

       >>> trace = sampler.run(1_000, compile=True)

    Behind the scene MCX replaces the for loop used internally (so it can
    display a progress bar) by JAX's `lax.scan`. The difference is most obvious
    for large models and number of samples.

    Finally, note that since the runtime keeps track of the last state it is possible
    to switch between the different execution models during sampling.

    """

    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        model: mcx.model,
        evaluator,
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
        5. Get initial positions from the evaluator.
        6. Get a function that returns a kernel given its parameters from the
        evaluator.

        Parameters
        ----------
        rng_key
            The key passed to JAX's random number generator. The runtime is
            in charge of splitting the key as it is being used.
        model
            The model whose posterior we want to sample.
        evaluator
            The evaluator that will be used to sampler the posterior.
        num_chains
            The number of chains that will be used concurrently for sampling.
        observations
            The variables we condition on and their values.

        Returns
        -------
        A sampler object.

        """
        validate_conditioning_variables(model, **observations)

        print("sampler: build the loglikelihood")
        transformed_model = evaluator.transform(model)
        loglikelihood = build_loglikelihood(transformed_model, **observations)
        loglikelihood_contributions = build_loglikelihoods(model, **observations)

        print("sampler: find the initial states")
        initial_positions, unravel_fn = get_initial_position(
            rng_key, model, num_chains, **observations
        )
        loglikelihood = flatten_loglikelihood(loglikelihood, unravel_fn)
        initial_state = evaluator.states(initial_positions, loglikelihood)

        print("sampler: build and compile the inference kernel")
        kernel_factory = evaluator.kernel_factory(loglikelihood)

        self.is_warmed_up = False
        self.rng_key = rng_key
        self.num_chains = num_chains
        self.evaluator = evaluator
        self.kernel_factory = kernel_factory
        self.state = initial_state
        self.unravel_fn = unravel_fn
        self.loglikelihood_contributions = loglikelihood_contributions

    def __iter__(self) -> Iterator:
        """Make the sampler behave like an iterable and an iterator.

        By making the sampler an iterable we can make it return samples
        in a for loop or in a list comprehensioon:

            >>> # Beware, these will loop indefinitely
            ... for sample in sampler:
            ...     print(sample)
            ...
            ... samples = [sample for sample in samples]


        If the sampler has not been warmed up before using it as an iterable
        it will warm up with a default of 1,000 steps. To use a different number
        of steps (say 100) and different warmup options, call the `warmup` method
        first:

            >>> sampler.warmup(100, compile=True)
            ... for sample in sampler:
            ...     print(sample)

        Note
        ----
        `update_chains` will be JIT-compiled every time the iterator is
        initialized. Just watch for the overhead with big models.

        Returns
        -------
        The current instance of the sampler as an iterable.

        """
        default_warmup_steps = 1_000
        if not self.is_warmed_up:
            _ = self.warmup(default_warmup_steps, compile=False)

        @jax.jit
        def update_chains(rng_key, parameters, state):
            kernel = self.kernel_factory(*parameters)
            new_states, info = kernel(rng_key, state)
            return new_states, info

        def run(rng_key, state, parameters):
            while True:
                _, rng_key = jax.random.split(rng_key)
                keys = jax.random.split(rng_key, self.num_chains)
                state, info = jax.vmap(update_chains, in_axes=(0, 0, 0))(
                    keys, parameters, state
                )
                yield (state, info)

        self.sample_generator = run(self.rng_key, self.state, self.parameters)

        return self

    def __next__(self) -> Tuple:
        """Yield the next state of the chain.

        This method allows the sampler to behave as an iterator:

            >>> iter(sampler)
            ... sample = next(sampler)

        Returns
        -------
        A tuple that contains the next sample from the chain and the
        corresponding sampling information.

        """
        new_state = next(self.sample_generator)
        self.state, info = new_state
        sample, sampling_info = self.evaluator.make_trace(
            chain=new_state, ravel_fn=self.unravel_fn
        )
        return sample, sampling_info

    def warmup(self, num_warmup_steps: int = 1000, compile: bool = False, **kwargs):
        """Warmup the sampler.

        Warmup is necessary to get values for the evaluator's parameters that are
        adapted to the geometry of the posterior distribution. While run will
        automatically run the warmup if it hasn't been done before, runnning
        this method independently gives access to the trace for the warmup
        phase and the values of the parameters for diagnostics.

        Parameters
        ----------
        num_warmup_steps
            The number of warmup_steps to perform.
        progress_bar
            If True the progress of the warmup will be displayed. Otherwise it
            will use `lax.scan` to iterate (which is potentially faster).
        kwargs
            Parameters to pass to the evaluator's warmup.

        Returns
        -------
        trace
            A Trace object that contains the warmup sampling chain, warmup sampling info
            and warmup info.

        """
        last_state, parameters, warmup_chain = self.evaluator.warmup(
            self.rng_key,
            self.state,
            self.kernel_factory,
            self.num_chains,
            num_warmup_steps,
            compile,
            **kwargs,
        )
        self.state = last_state
        self.parameters = parameters
        self.is_warmed_up = True

        # The evaluator must return `None` when no warmup is needed.
        if warmup_chain is None:
            return

        samples, sampling_info, warmup_info = self.evaluator.make_warmup_trace(
            chain=warmup_chain, unravel_fn=self.unravel_fn
        )

        trace = Trace(
            warmup_samples=samples,
            warmup_sampling_info=sampling_info,
            warmup_info=warmup_info,
            loglikelihood_contributions_fn=self.loglikelihood_contributions,
        )

        return trace

    def run(
        self,
        num_samples: int = 1000,
        num_warmup_steps: int = 1000,
        compile: bool = False,
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
        compile
            If False the progress of the warmup and samplig will be displayed.
            Otherwise it will use `lax.scan` to iterate (which is potentially
            faster).
        warmup_kwargs
            Parameters to pass to the evaluator's warmup.

        Returns
        -------
        trace
            A Trace object that contains the chains, some information about
            the inference process (e.g. divergences for evaluators in the
            HMC family).

        """
        if not self.is_warmed_up:
            self.warmup(num_warmup_steps, compile, **warmup_kwargs)

        @jax.jit
        def update_chain(rng_key, parameters, chain_state):
            kernel = self.kernel_factory(*parameters)
            new_chain_state, info = kernel(rng_key, chain_state)
            return new_chain_state, info

        _, self.rng_key = jax.random.split(self.rng_key)
        rng_keys = jax.random.split(self.rng_key, num_samples)

        # The progress bar, displayed when compile=False, is an important
        # indicator for exploratory analysis, while sample_scan is optimal for
        # getting a large number of samples.  Note that for small sample sizes
        # lax.scan is not dramatically faster than a for loop due to compile
        # time. Expect however an increasing improvement as the number of
        # samples increases.
        if compile:
            last_state, chain = sample_scan(
                update_chain, self.state, self.parameters, rng_keys, self.num_chains
            )
        else:
            last_state, chain = sample_loop(
                update_chain, self.state, self.parameters, rng_keys, self.num_chains
            )

        samples, sampling_info = self.evaluator.make_trace(
            chain=chain, unravel_fn=self.unravel_fn
        )
        trace = Trace(
            samples=samples,
            sampling_info=sampling_info,
            loglikelihood_contributions_fn=self.loglikelihood_contributions,
        )

        self.state = last_state

        return trace


#
# SHARED UTILITIES
#


def sample_scan(
    kernel: Callable,
    init_state: np.DeviceArray,
    parameters: np.DeviceArray,
    rng_keys: np.DeviceArray,
    num_chains: int,
) -> Tuple:
    """Sample using JAX's scan.

    Using scan over other constructs (and python loops) means reduced compilation time
    for loops with many iterations. Since the loop itself is compiled, sampling with
    scan will be faster than with the for loop. We thus recommend to use `sample_scan`
    whenever possible.

    Parameters
    ----------
    kernel
        The function that will be repeatedly applied to the chain state. It
        must include the transition kernel, but can also compute online
        diagnostics.
    init_state: array (n_chains, n_dim)
        The initial chain state.
    parameters: array (n_chains, n_parameters)
        The parameters of the evaluator.
    rng_keys: array (n_samples,)
        JAX PRNGKeys used for each sampling step.
    num_chains
        The number of chains

    Returns
    -------
    The last state of the chain as well as the full chain.

    """
    num_samples = rng_keys.shape[0]

    print(f"Draw {num_samples:,} samples from {num_chains:,} chains.", end=" ")
    start = datetime.now()

    @jax.jit
    def update_scan(carry, key):
        state, parameters = carry
        keys = jax.random.split(key, num_chains)
        state, info = jax.vmap(kernel, in_axes=(0, 0, 0))(keys, parameters, state)
        return (state, parameters), (state, info)

    last_state, chain = jax.lax.scan(update_scan, (init_state, parameters), rng_keys)
    last_chain_state = last_state[0]

    mcx.jax.wait_until_computed(chain)
    mcx.jax.wait_until_computed(last_state)
    print(f"Done in {(datetime.now()-start).total_seconds():.1f} s.")

    return last_chain_state, chain


def sample_loop(
    kernel: Callable,
    init_state: np.DeviceArray,
    parameters: np.DeviceArray,
    rng_keys: np.DeviceArray,
    num_chains: int,
) -> Tuple:
    """Sample using a Python loop.

    While `sample_scan` is more performant it may make sense to accept slower
    sampling during the initial model iteration to get an idea of the speed at
    which the sampler works and get regular updates on the diagnostics.

    Note
    ----
    `jax.lax.scan` outputs a tuple (State, Info) where State and Info contain the
    information for each sample stacked. On the other hand, naive iterative
    sampling returns a list of (State, Info) tuples. The former being more
    convenient to build the trace we reshape the output of the `for` loop. This
    is slightly more complicated that we originally thought.

    While we can copy the implementation of `jax.lax.scan` (non-jitted version) to get
    the same output `jax.tree_util.tree_multimap(np.stack, chain)`, this is slow for large
    numbers of samples. Since the work happens outside of the progress bar this can be
    very confusing for the user.

    The next possibility is to ravel the pytree of the kernel's output, stack
    the ravelled pytrees, then unravel. This is made difficult by the fact that
    JAX's version of ravel_pytree uses `vjp` and therefore does not support
    some python types like booleans. We thus use Numpyro's `ravel_pytree`
    function to ravel while sampling. Since stacking the ravelled pytrees and
    unravelling is fast, this solution is much faster and shows the user a
    realistic number of samples per second. Sampling speed is clearly limited
    by I/O consideration, efforts to speed things up should be focused on this.

    Parameters
    ----------
    kernel
        The function that will be repeatedly applied to the chain state. It
        must include the transition kernel, but can also compute online
        diagnostics.
    init_state: array (n_chains, n_dim)
        The initial chain state.
    parameters: array (n_chains, n_parameters)
        The parameters of the evaluator.
    rng_keys: array (n_samples,)
        JAX PRNGKeys used for each sampling step.
    num_chains
        The number of chains

    Returns
    -------
    The last state of the chain as well as the full chain.

    """
    num_samples = rng_keys.shape[0]

    @jax.jit
    def update_loop(state, key):
        keys = jax.random.split(key, num_chains)
        state, info = jax.vmap(kernel, in_axes=(0, 0, 0))(keys, parameters, state)
        return state, info, mcx_ravel_pytree((state, info))[0]

    # we get the unraveling function for the tuple (state, info) by doing a dry
    # run.
    def get_unravel_fn():
        state, info, _ = update_loop(init_state, rng_keys[0])
        return mcx_ravel_pytree((state, info))

    _, unravel_fn = get_unravel_fn()

    with tqdm(rng_keys, unit="samples", mininterval=1) as progress:
        progress.set_description(
            f"Collecting {num_samples:,} samples across {num_chains:,} chains",
            refresh=False,
        )
        chain = []
        state = init_state
        for i, key in enumerate(progress):
            state, _, ravelled_state = update_loop(state, key)
            chain.append(ravelled_state)

    chain = np.stack(chain)
    chain = jax.vmap(unravel_fn)(chain)
    last_state = state

    return last_state, chain


def validate_conditioning_variables(model, **kwargs):
    """Check that all variables passed as arguments to the sampler
    are random variables or arguments to the sampler. And converserly
    that all of the model definition's positional arguments are given
    a value.

    TODO: Move to model.py?
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
            f"You passed a value for {unknown_str} which are neither random variables "
            "nor arguments to the model definition."
        )

    # The user must provide a value for all of the model definition's
    # positional arguments.
    model_posargs = set(model.posargs)
    if model_posargs.difference(conditioning_vars):
        missing_vars = model_posargs.difference(conditioning_vars)
        missing_str = ", ".join(missing_vars)
        raise AttributeError(
            f"You need to specify a value for the following arguments: {missing_str}"
        )


def build_loglikelihood(model, **kwargs):
    artifact = compile_to_logpdf(model.graph, model.namespace)
    logpdf = artifact.compiled_fn
    loglikelihood = jax.partial(logpdf, **kwargs)
    return loglikelihood


def build_loglikelihoods(model, **kwargs):
    """Function to compute the loglikelihood contribution
    of each variable.
    """
    artifact = compile_to_loglikelihoods(model.graph, model.namespace)
    logpdf = artifact.compiled_fn
    loglikelihoods = jax.partial(logpdf, **kwargs)
    return loglikelihoods


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
    _, unravel_fn = jax_ravel_pytree(sample_position_dict)

    return positions, unravel_fn


def flatten_loglikelihood(logpdf, unravel_fn):
    def flattened_logpdf(array):
        kwargs = unravel_fn(array)
        return logpdf(**kwargs)

    return flattened_logpdf
