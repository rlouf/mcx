import jax


def sample(rng_key, runtime, num_samples=1000, num_warmup=1000, num_chains=4, **kwargs):
    """ The sampling runtime. """
    initialize, build_kernel, to_trace = runtime

    loglikelihood, initial_state, parameters, unravel_fn = initialize(rng_key, num_chains, **kwargs)
    loglikelihood = jax.jit(loglikelihood)

    kernel = build_kernel(loglikelihood, parameters)
    kernel = jax.jit(kernel)

    # Run the inference
    @jax.jit
    def update_chains(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        new_states, info = jax.vmap(kernel, in_axes=(0, 0))(keys, states)
        return new_states, states

    rng_keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(update_chains, initial_state, rng_keys)

    trace = to_trace(states, unravel_fn)

    return trace


def generate(rng_key, runtime, num_warmup=1000, num_chains=4, **kwargs):
    """ The generator runtime """

    initialize, build_kernel, to_trace = runtime

    loglikelihood, initial_state, parameters, unravel_fn = initialize(rng_key, num_chains, **kwargs)
    loglikelihood = jax.jit(loglikelihood)

    kernel = build_kernel(loglikelihood, parameters)
    kernel = jax.jit(kernel)

    state = initial_state
    while True:
        _, rng_key = jax.random.split(rng_key)

        keys = jax.random.split(rng_key, num_chains)
        new_states = jax.vmap(kernel)(keys, state)

        yield new_states
