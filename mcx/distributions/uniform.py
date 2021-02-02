from jax import numpy as jnp
from jax import random

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class Uniform(Distribution):
    parameters = {"lower": constraints.real, "upper": constraints.real}

    def __init__(self, lower, upper):
        self.support = constraints.closed_interval(lower, upper)

        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(jnp.shape(lower), jnp.shape(upper))
        self.lower = lower
        self.upper = upper

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        u = random.uniform(rng_key, shape)
        return u * (self.upper - self.lower) + self.lower

    @constraints.limit_to_support
    def logpdf(self, x):
        return jnp.log(1.0 / (self.upper - self.lower))
