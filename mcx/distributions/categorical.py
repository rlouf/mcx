from jax import numpy as jnp
from jax import random
from jax.scipy.special import xlogy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class Categorical(Distribution):
    parameters = {"probs": constraints.simplex}

    def __init__(self, probs):
        self.support = constraints.integer_interval(0, jnp.shape(probs)[-1] - 1)
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(jnp.shape(probs)[:-1])
        self.probs = probs

    def sample(self, rng_key, sample_shape):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.categorical(rng_key, self.probs, axis=-1, shape=shape)

    @constraints.limit_to_support
    def logpdf(self, x):
        x_array = jnp.arange(self.probs.shape[-1]) == x
        return jnp.sum(xlogy(x_array, self.probs), axis=-1)
