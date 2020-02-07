from jax import numpy as np
from jax import random
from jax.scipy import stats

from . import constraints
from .distribution import Distribution
from .utils import broadcast_batch_shape, limit_to_support


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

    @limit_to_support
    def logpdf(self, x):
        return stats.laplace(x, loc=self.loc, scale=self.scale)
