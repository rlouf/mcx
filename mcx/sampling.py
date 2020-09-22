from typing import Callable, Tuple

import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from tqdm import tqdm

import mcx
from mcx import sample_forward
from mcx.core import compile_to_logpdf


__all__ = ["sample", "generate", "sequential"]


# -------------------------------------------------------------------
#                 == THE SAMPLING EXECUTION MODEL ==
# -------------------------------------------------------------------


class sample(object):
    def __init__(
        self,
        rng_key: jax.random.PRNGKey,
        model: mcx.model,
        program,
        num_warmup_steps: int = 1000,
        num_chains: int = 4,
        **kwargs
    ):
        """Initialize the sampling runtime."""
        self.program = program
        self.num_chains = num_chains
        self.rng_key = rng_key

        init, warmup, build_kernel, to_trace, adapt_loglikelihood = self.program

        print("Initialize the sampler\n")
        validate_conditioning_variables(model, **kwargs)
        loglikelihood = build_loglikelihood(model, **kwargs)

        print("Find initial states...")
        initial_position, unravel_fn = get_initial_position(
            rng_key, model, num_chains, **kwargs
        )
        loglikelihood = flatten_loglikelihood(loglikelihood, unravel_fn)
        loglikelihood = adapt_loglikelihood(loglikelihood)
        initial_state = jax.vmap(init, in_axes=(0, None))(
            initial_position, jax.value_and_grad(loglikelihood)
        )

        print("Warmup the chains...")
        parameters, state = warmup(
            rng_key, initial_state, loglikelihood, num_warmup_steps
        )

        print("Compile the log-likelihood...")
        loglikelihood = jax.jit(loglikelihood)

        print("Build and compile the inference kernel...")
        # We would like to be able to JIT-compile the kernel
        # builder. However there is too much overlapping in HMC's namespace
        # and it looks like the function might be changing a global state.
        # TODO: Fix this leakeage to be able to compile the function.
        kernel_builder = build_kernel(loglikelihood)
        kernel_builder = kernel_builder

        self.kernel_builder = kernel_builder
        self.parameters = parameters
        self.state = state
        self.to_trace = to_trace
        self.unravel_fn = unravel_fn

    def run(self, num_samples=1000):
        _, self.rng_key = jax.random.split(self.rng_key)

        @jax.jit
        def update_chains(rng_key, parameters, state):
            kernel = self.kernel_builder(parameters)
            new_states, info = kernel(rng_key, state)
            return new_states, info

        state = self.state
        chain = []

        rng_keys = jax.random.split(self.rng_key, num_samples)
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
        self.state = state

        trace = self.to_trace(chain, self.unravel_fn)

        return trace


# -------------------------------------------------------------------
#                 == THE GENERATOR EXECUTION MODEL ==
# -------------------------------------------------------------------


def generate(rng_key, model, program, num_warmup_steps=1000, num_chains=4, **kwargs):
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
                "Collecting {:,} samples".format(self.num_samples), refresh=False,
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

    sample_position_dict = {
        parameter: values[0] for parameter, values in initial_positions.items()
    }
    _, unravel_fn = ravel_pytree(sample_position_dict)

    return positions, unravel_fn


def flatten_loglikelihood(logpdf, unravel_fn):
    def flattened_logpdf(array):
        kwargs = unravel_fn(array)
        return logpdf(**kwargs)

    return flattened_logpdf
