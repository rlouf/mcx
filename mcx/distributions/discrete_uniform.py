from jax import lax
from jax import numpy as jnp
from jax import random

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class DiscreteUniform(Distribution):
    """Random variable with a uniform distribution on a range of integers.  """

    parameters = {"lower": constraints.integer, "upper": constraints.integer}

    def __init__(self, lower, upper):
        self.support = constraints.integer_interval(lower, upper)

        self.event_shape = ()
        lower, upper = promote_shapes(lower, upper)
        batch_shape = lax.broadcast_shapes(jnp.shape(lower), jnp.shape(upper))
        self.batch_shape = batch_shape
        self.lower = jnp.broadcast_to(jnp.floor(lower), batch_shape)
        self.upper = jnp.broadcast_to(jnp.floor(upper), batch_shape)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.randint(rng_key, shape, self.lower, self.upper)

    @constraints.limit_to_support
    def logpdf(self, x):
        return -jnp.log(self.upper - self.lower + 1)
