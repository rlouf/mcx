from jax import random
from jax.scipy import stats

from . import constraints
from .distribution import Distribution
from .utils import broadcast_batch_shape, limit_to_support


class Beta(Distribution):
    parameters = {
        "a": constraints.strictly_positive,
        "b": constraints.strictly_positive,
    }
    support = constraints.interval(0, 1)

    def __init__(self, a, b):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(a, b)
        self.a = a
        self.b = b

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.beta(rng_key, self.a, self.b, shape=shape)

    @limit_to_support
    def logpdf(self, x):
        return stats.beta.logpdf(x, self.a, self.b)
