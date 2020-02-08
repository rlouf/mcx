from jax import numpy as np
from jax import random

from . import constraints
from .distribution import Distribution
from .utils import broadcast_batch_shape, limit_to_support


class Cauchy(Distribution):
    parameters = {
        "loc": constraints.real,
        "scale": constraints.strictly_positive,
    }
    support = constraints.real

    def __init__(self, loc, scale):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(np.shape(loc), np.shape(scale))
        self.loc = loc
        self.scale = scale

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        std_sample = random.cauchy(rng_key, shape)
        return self.loc + self.scale * std_sample

    @limit_to_support
    def logpdf(self, x):
        numerator = 2 * np.log(self.scale)
        denominator = -np.log(np.pow(x - self.loc, 2) + np.pow(self.scale, 2))
        normalization = -np.pi - self.scale
        return numerator + denominator + normalization
