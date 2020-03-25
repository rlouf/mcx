import jax


class sample(object):

    def __init__(self, rng_key, runtime, num_warmup=1000, num_chains=4, **kwargs):
        """ Initialize the sampling runtime.
        """
        self.runtime = runtime
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.rng_key = rng_key

        initialize, build_kernel, to_trace = self.runtime
        loglikelihood, initial_state, parameters, unravel_fn = initialize(rng_key, self.num_chains, **kwargs)
        loglikelihood = jax.jit(loglikelihood)

        kernel = build_kernel(loglikelihood, parameters)
        self.kernel = jax.jit(kernel)

        self.state = initial_state
        self.unravel_fn = unravel_fn
        self.to_trace = to_trace

    def take(self, num_samples=1000):
        _, self.rng_key = jax.random.split(self.rng_key)

        @jax.jit
        def update_chains(states, rng_key):
            keys = jax.random.split(rng_key, self.num_chains)
            new_states, info = jax.vmap(self.kernel, in_axes=(0, 0))(keys, states)
            return new_states, states

        rng_keys = jax.random.split(self.rng_key, num_samples)
        last_state, states = jax.lax.scan(update_chains, self.state, rng_keys)
        self.state = last_state

        trace = self.to_trace(states, self.unravel_fn)

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
