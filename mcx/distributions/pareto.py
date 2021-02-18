from jax import numpy as np
from jax import random
from jax.scipy import stats

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class Pareto(Distribution):
    parameters = {
        "shape": constraints.strictly_positive,
        "scale": constraints.strictly_positive,
    }

    def __init__(self, shape, scale, loc=0):
        self.support = constraints.closed_interval(scale, np.inf)
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(np.shape(shape), np.shape(scale))
        self.shape = shape
        self.scale = scale
        self.loc = loc

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return self.scale * (random.pareto(key=rng_key, b=self.shape, shape=shape))

    @constraints.limit_to_support
    def logpdf(self, x):
        return stats.pareto.logpdf(x=x, b=self.shape, loc=self.loc, scale=self.scale)
