import time

import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from tqdm import tqdm

from mcx import sample_forward
from mcx.core import compile_to_logpdf


class sample(object):
    def __init__(
        self, rng_key, model, program, num_warmup=1000, num_chains=4, **kwargs
    ):
        """ Initialize the sampling runtime.
        """
        self.program = program
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.rng_key = rng_key

        print("Initialize the sampler")
        print("----------------------")

        init, warmup, build_kernel, to_trace = self.program

        check_conditioning_variables(model, **kwargs)
        loglikelihood = build_loglikelihood(model, **kwargs)

        print("Find initial positions...", end=" ")
        start = time.time()
        positions, unravel_fn = get_initial_position(
            rng_key, model, num_chains, **kwargs
        )
        print("Found in {:.2f}s".format(time.time() - start))

        flat_loglikelihood = _flatten_logpdf(loglikelihood, unravel_fn)

        print("Compute initial states...", end=" ")
        start = time.time()
        value_and_grad = jax.value_and_grad(flat_loglikelihood)
        initial_state = jax.vmap(init, in_axes=(0, None))(positions, value_and_grad)
        print("Computed in {:.2f}s".format(time.time() - start))

        parameters, state = warmup(initial_state, flat_loglikelihood)

        print("Compiling the log-likelihood...", end=" ")
        start = time.time()
        loglikelihood = jax.jit(flat_loglikelihood)
        print("Compiled in {:.2f}s".format(time.time() - start))

        print("Compiling the inference kernel...", end=" ")
        start = time.time()
        kernel = build_kernel(loglikelihood, parameters)
        kernel = jax.jit(kernel)
        print("Compiled in {:.2f}s".format(time.time() - start))

        self.kernel = kernel
        self.state = state
        self.to_trace = to_trace
        self.unravel_fn = unravel_fn

    def run(self, num_samples=1000):
        _, self.rng_key = jax.random.split(self.rng_key)

        @jax.jit
        def update_chains(state, rng_key):
            keys = jax.random.split(rng_key, self.num_chains)
            new_states, info = jax.vmap(self.kernel, in_axes=(0, 0))(keys, state)
            return new_states

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
                state = update_chains(state, key)
                chain.append(state)
        self.state = state

        trace = self.to_trace(chain, self.unravel_fn)

        return trace


def generate(rng_key, runtime, num_warmup=1000, num_chains=4, **kwargs):
    """ The generator runtime """

    initialize, build_kernel, to_trace = runtime

    loglikelihood, initial_state, parameters, unravel_fn = initialize(
        rng_key, num_chains, **kwargs
    )
    loglikelihood = jax.jit(loglikelihood)

    kernel = build_kernel(loglikelihood, parameters)
    kernel = jax.jit(kernel)

    state = initial_state
    while True:
        _, rng_key = jax.random.split(rng_key)

        keys = jax.random.split(rng_key, num_chains)
        new_states = jax.vmap(kernel)(keys, state)

        yield new_states


def check_conditioning_variables(model, **kwargs):
    """ Check that all variables passed as arguments to the sampler
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

    positions = []
    for i in range(num_chains):
        position = {k: value[i] for k, value in initial_positions.items()}
        flat_position, unravel_fn = ravel_pytree(position)
        positions.append(flat_position)
    positions = np.stack(positions)

    return positions, unravel_fn


def _flatten_logpdf(logpdf, unravel_fn):
    def flattened_logpdf(array):
        kwargs = unravel_fn(array)
        return logpdf(**kwargs)

    return flattened_logpdf
