from jax import numpy as np
from jax import random

from . import constraints
from .distribution import Distribution
from .utils import broadcast_batch_shape, limit_to_support


class DiscreteUniform(Distribution):
    """Random variable with a uniform distribution on a range of integers.  """

    parameters = {"lower": constraints.integer, "upper": constraints.integer}

    def __init__(self, lower, upper):
        self.support = constraints.integer_interval(lower, upper)

        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(np.shape(lower), np.shape(upper))
        self.lower = np.floor(lower)
        self.upper = np.floor(upper)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.randint(rng_key, shape, self.lower, self.upper)

    @limit_to_support
    def logpdf(self, x):
        return -np.log(self.upper - self.lower + 1)
