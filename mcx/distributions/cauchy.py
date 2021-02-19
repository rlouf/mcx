from jax import numpy as jnp
from jax import random, scipy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class Cauchy(Distribution):
    parameters = {
        "loc": constraints.real,
        "scale": constraints.strictly_positive,
    }
    support = constraints.real

    def __init__(self, loc, scale):
        self.event_shape = ()
        batch_shape, (loc, scale) = promote_shapes(loc, scale)
        self.batch_shape = batch_shape
        self.loc = jnp.broadcast_to(loc, batch_shape)
        self.scale = jnp.broadcast_to(scale, batch_shape)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        std_sample = random.cauchy(rng_key, shape)
        return self.loc + self.scale * std_sample

    @constraints.limit_to_support
    def logpdf(self, x):
        return scipy.stats.cauchy.logpdf(x, self.loc, self.scale)
