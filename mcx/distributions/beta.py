from jax import numpy as jnp
from jax import random
from jax.scipy import stats

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class Beta(Distribution):
    parameters = {
        "a": constraints.strictly_positive,
        "b": constraints.strictly_positive,
    }
    support = constraints.interval(0, 1)

    def __init__(self, a, b):
        self.event_shape = ()
        batch_shape, (a, b) = promote_shapes(a, b)
        self.batch_shape = batch_shape
        self.a = jnp.broadcast_to(a, batch_shape)
        self.b = jnp.broadcast_to(b, batch_shape)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.beta(rng_key, self.a, self.b, shape=shape)

    @constraints.limit_to_support
    def logpdf(self, x):
        return stats.beta.logpdf(x, self.a, self.b)
