from jax import numpy as np
from jax import random
from jax.scipy.special import xlog1py, xlogy

from . import constraints
from .distribution import Distribution
from .utils import broadcast_batch_shape, limit_to_support


class Bernoulli(Distribution):
    parameters = {"p": constraints.probability}
    support = constraints.boolean

    def __init__(self, p):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(np.shape(p))
        self.p = p * 1.0  # will fail if p is int

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.bernoulli(rng_key, self.p, shape=shape)

    @limit_to_support
    def logpdf(self, x):
        """ (TODO): Check that x belongs to support, return -infty otherwise
        """
        return xlogy(x, self.p) + xlog1py(1 - x, -self.p)
