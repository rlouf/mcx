from jax import numpy as np
from jax import random
from jax.scipy import stats

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class Pareto(Distribution):
    parameters = {
        "b": constraints.strictly_positive,
    }

    def __init__(self, a, m):
        self.support = constraints.closed_interval(m, np.inf)
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(np.shape(a), np.shape(m))
        self.a = a
        self.m = m

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.pareto(key=rng_key, b=self.b, shape=shape)

    @constraints.limit_to_support
    def logpdf(self, x):
        return stats.pareto.logpdf(x=x, b=self.b, loc=self.m)
