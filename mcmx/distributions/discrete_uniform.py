from jax import numpy as np
from jax import random

from . import constraints
from .distribution import Distribution
from .utils import broadcast_batch_shape


class DiscreteUniform(Distribution):
    """Random variable with a uniform distribution on a range of integers.  """

    params_constraints = {"lower": constraints.integer, "upper": constraints.integer}

    def __init__(self, lower, upper):
        self.suport = constraints.integer_interval(self.lower, self.upper)

        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(lower, upper)
        self.lower = np.floor(lower)
        self.upper = np.floor(upper)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.randint(rng_key, shape, self.lower, self.upper)

    def logpdf(self, x):
        return -np.log(self.upper - self.lower - 1)
