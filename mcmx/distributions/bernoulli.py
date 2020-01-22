from jax import random
from jax.scipy.special import xlogy, xlog1py

from .distribution import Distribution
from . import constraints
from .utils import broadcast_batch_shape


class Bernoulli(Distribution):
    param_constraints = {"p": constraints.probability}
    support = constraints.boolean

    def __init__(self, p):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(p)
        self.p = p

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.bernoulli(rng_key, self.p, shape=shape)

    def logpdf(self, k):
        return xlogy(k, self.p) + xlog1py(k - 1, -self.p)
