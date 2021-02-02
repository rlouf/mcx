from jax import numpy as jnp
from jax import random

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class DiscreteUniform(Distribution):
    """Random variable with a uniform distribution on a range of integers.  """

    parameters = {"lower": constraints.integer, "upper": constraints.integer}

    def __init__(self, lower, upper):
        self.support = constraints.integer_interval(lower, upper)

        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(jnp.shape(lower), jnp.shape(upper))
        self.lower = jnp.floor(lower)
        self.upper = jnp.floor(upper)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.randint(rng_key, shape, self.lower, self.upper)

    @constraints.limit_to_support
    def logpdf(self, x):
        return -jnp.log(self.upper - self.lower + 1)
