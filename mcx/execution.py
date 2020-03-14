import jax
import jax.numpy as np

from mcx.inference.runtimes.runtime import Runtime


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
