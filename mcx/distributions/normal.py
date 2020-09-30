from jax import numpy as np
from jax import random, scipy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class Normal(Distribution):
    params_constraints = {
        "mu": constraints.real,
        "sigma": constraints.strictly_positive,
    }
    support = constraints.real

    def __init__(self, mu, sigma):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(np.shape(mu), np.shape(sigma))
        self.mu = mu
        self.sigma = sigma
        super(Normal, self).__init__()

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        std_sample = random.normal(rng_key, shape=shape)
        return self.mu + self.sigma * std_sample

    # no need to check on support ]-infty, +infty[
    def logpdf(self, x):
        return scipy.stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)
