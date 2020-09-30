from jax import numpy as np
from jax import random
from jax.scipy import stats

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class Laplace(Distribution):
    parameters = {"loc": constraints.real, "scale": constraints.strictly_positive}
    support = constraints.real

    def __init__(self, loc, scale):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(np.shape(loc), np.shape(scale))
        self.loc = loc
        self.scale = scale

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return self.loc + random.laplace(rng_key, self.scale, shape)

    @constraints.limit_to_support
    def logpdf(self, x):
        return stats.laplace(x, loc=self.loc, scale=self.scale)
