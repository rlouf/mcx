import jax
import jax.numpy as np

from mcx.inference.runtime import Runtime


def sample(runtime: Runtime, num_samples=1000, num_warmup=1000, num_chains=4, **kwargs):
    """Sample is a runtime: it is in charge of executing the program and manage
    the input (data) - output.
    """
    # Check that all necessary variables were passed (it will yell anyway)
    logpdf = runtime.logpdf_fn
    logpdf = jax.partial(logpdf, **kwargs)
    logpdf = jax.jit(logpdf)
    initial_state = runtime.state
    if runtime.state is None:
        initial_state = runtime.initialize(logpdf, num_warmup, num_chains, **kwargs)

    # Create and compile the inference kernel
    kernel = runtime.inference_kernel(
        logpdf, np.array([1.0, 1.0]), np.array([1.0, 1.0])
    )
    kernel = jax.jit(kernel)

    # Run the inference
    @jax.jit
    # @jax.jit
    def chain_update(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        new_states = jax.vmap(kernel)(keys, states)
        new_states = jax.vmap(kernel, in_axes=(0, 0))(keys, states)
        return new_states, new_states

    # scanning over vectorization is likely to be completely inefficient!
    # we may circumvent this by running a "test" chain in parallel in a
    # fori loop to return divergences?
    rng_keys = jax.random.split(runtime.rng_key, num_samples)
    states = jax.lax.scan(chain_update, initial_state, rng_keys)

    trace = runtime.to_trace(states)

    return trace


def generate(runtime: Runtime, num_warmup=1000, num_chains=4, **kwargs):
    """Returns a generator of samples.
    """

    # Check that all necessary variables were passed (it will yell anyway)
    logpdf = runtime.logpdf_fn
    logpdf = jax.partial(logpdf, **kwargs)
    logpdf = jax.jit(logpdf)

    initial_state = runtime.state
    if runtime.state is None:
        initial_state = runtime.initialize(logpdf, num_warmup, num_chains, **kwargs)

    # Create and compile the inference kernel
    kernel = runtime.inference_kernel(
        logpdf, np.array([1.0, 1.0]), np.array([1.0, 1.0]),
    )
    kernel = jax.jit(kernel)

    states = initial_state
    rng_key = runtime.rng_key
    while True:
        _, rng_key = jax.random.split(rng_key)

        keys = jax.random.split(rng_key, num_chains)
        new_states = jax.vmap(kernel)(keys, states)

        yield new_states
