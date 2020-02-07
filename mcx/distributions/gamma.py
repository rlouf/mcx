from jax import random
from jax.scipy import stats

from . import constraints
from .distribution import Distribution
from .utils import broadcast_batch_shape, limit_to_support


class Gamma(Distribution):
    parameters = {
        "a": constraints.strictly_positive,
        "loc": constraints.real,
        "scale": constraints.strictly_positive,
    }
    support = constraints.strictly_positive

    def __init__(self, a, loc, scale):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(a, loc, scale)
        self.a = a
        self.loc = loc
        self.scale = scale

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.gamma(rng_key, self.a, self.loc, self.scale, shape)

    @limit_to_support
    def logpdf(self, x):
        return stats.gamma(x, self.a, self.loc, self.scale)
