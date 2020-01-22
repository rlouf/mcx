from jax import random

from . import constraints
from .distribution import Distribution
from .utils import broadcast_batch_shape


class Uniform(Distribution):
    params_constraints = {"lower": constraints.real, "upper": constraints.real}

    def __init__(self, lower, upper):
        self.support = constraints.interval(self.lower, self.upper)

        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(lower, upper)
        self.lower = lower
        self.upper = upper

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        u = random.uniform(rng_key, shape)
        return u * (self.upper - self.lower) + self.lower

    def logpdf(self, x):
        return 1.0 / (self.upper - self.lower)
