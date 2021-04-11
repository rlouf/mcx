from jax import numpy as jnp
from jax import random, scipy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution


class Poisson(Distribution):
    parameters = {"lambda": constraints.positive}
    support = constraints.whole_number

    def __init__(self, lmbda):
        self.event_shape = ()
        self.batch_shape = jnp.shape(lmbda)
        self.lmbda = lmbda

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.poisson(rng_key, self.lmbda, shape)

    @constraints.limit_to_support
    def logpdf(self, k):
        return scipy.stats.poisson.logpmf(k, self.lmbda)
