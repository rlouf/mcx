from jax import numpy as jnp
from jax import random

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class LogNormal(Distribution):
    parameters = {"mu": constraints.real, "sigma": constraints.positive}
    support = constraints.strictly_positive

    def __init__(self, mu, sigma):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(jnp.shape(mu), jnp.shape(sigma))
        self.mu = mu
        self.sigma = sigma

    def sample(self, rng_key, sample_shape):
        shape = sample_shape + self.batch_shape + self.event_shape
        return jnp.exp(self.sigma * random.normal(rng_key, shape) + self.mu)

    @constraints.limit_to_support
    def logpdf(self, x):
        return -((jnp.log(x) - self.mu) ** 2 / (2 * self.sigma ** 2)) - jnp.log(
            self.sigma * x * jnp.sqrt(2 * jnp.pi)
        )
