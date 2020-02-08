from jax import lax, random
from jax import numpy as np
from jax.scipy import stats

from . import constraints
from .distribution import Distribution
from .utils import broadcast_batch_shape, limit_to_support


class Exponential(Distribution):
    parameters = {"lmbda": constraints.strictly_positive}
    support = constraints.positive

    def __init__(self, lmbda):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(np.shape(lmbda))
        self.lmbda = lmbda

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.event_shape + self.batch_shape
        return random.exponential(rng_key, shape=shape) / self.lmbda

    @limit_to_support
    def logpdf(self, x):
        scale = lax.div(1.0, self.lmbda)
        return stats.exponential(x, loc=0.0, scale=scale)
