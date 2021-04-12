from jax import lax
from jax import numpy as jnp
from jax import random, scipy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution


class HalfNormal(Distribution):
    params_constraints = {
        "sigma": constraints.strictly_positive,
    }
    support = constraints.positive

    def __init__(self, sigma):
        self.event_shape = ()
        self.batch_shape = jnp.shape(sigma)
        self.sigma = sigma

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        std_sample = random.truncated_normal(rng_key, lower=0, upper=None, shape=shape)
        return self.sigma * std_sample

    @constraints.limit_to_support
    def logpdf(self, x):
        return lax.add(
            scipy.stats.norm.logpdf(x, loc=0, scale=self.sigma), lax.log(2.0)
        )
