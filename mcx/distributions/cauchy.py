from jax import numpy as jnp
from jax import random, scipy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class Cauchy(Distribution):
    parameters = {
        "loc": constraints.real,
        "scale": constraints.strictly_positive,
    }
    support = constraints.real

    def __init__(self, loc, scale):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(jnp.shape(loc), jnp.shape(scale))
        self.loc = loc
        self.scale = scale

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        std_sample = random.cauchy(rng_key, shape)
        return self.loc + self.scale * std_sample

    @constraints.limit_to_support
    def logpdf(self, x):
        return scipy.stats.cauchy.logpdf(x, self.loc, self.scale)
