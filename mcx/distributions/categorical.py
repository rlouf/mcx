from jax import numpy as np
from jax import random
from jax.scipy.special import xlogy

from . import constraints
from .distribution import Distribution
from .utils import limit_to_support


class Categorical(Distribution):
    parameters = {"probs": constraints.simplex}

    def __init__(self, probs):
        self.support = constraints.integer_interval(0, probs.shape[-1] - 1)

        self.event_shape = ()
        self.batch_shape = probs.shape[:-1]
        self.probs = probs

    def sample(self, rng_key, sample_shape):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.categorical(rng_key, self.probs, axis=0, shape=shape)

    @limit_to_support
    def logpdf(self, x):
        x_array = np.arange(self.probs.shape[-1]) == x
        return np.sum(xlogy(x_array, self.probs), axis=-1)
