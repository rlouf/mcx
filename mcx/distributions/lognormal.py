from jax import numpy as jnp
from jax import random

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class LogNormal(Distribution):
    parameters = {"mu": constraints.real, "sigma": constraints.positive}
    support = constraints.strictly_positive

    def __init__(self, mu, sigma):
        self.event_shape = ()
        batch_shape, (mu, sigma) = promote_shapes(mu, sigma)
        self.batch_shape = batch_shape
        self.mu = jnp.broadcast_to(mu, batch_shape)
        self.sigma = jnp.broadcast_to(sigma, batch_shape)

    def sample(self, rng_key, sample_shape):
        shape = sample_shape + self.batch_shape + self.event_shape
        return jnp.exp(self.sigma * random.normal(rng_key, shape) + self.mu)

    @constraints.limit_to_support
    def logpdf(self, x):
        return -((jnp.log(x) - self.mu) ** 2 / (2 * self.sigma ** 2)) - jnp.log(
            self.sigma * x * jnp.sqrt(2 * jnp.pi)
        )
