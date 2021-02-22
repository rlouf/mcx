from jax import lax
from jax import numpy as jnp
from jax import random, scipy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class Normal(Distribution):
    params_constraints = {
        "mu": constraints.real,
        "sigma": constraints.strictly_positive,
    }
    support = constraints.real

    def __init__(self, mu, sigma):
        self.event_shape = ()
        mu, sigma = promote_shapes(mu, sigma)
        batch_shape = lax.broadcast_shapes(jnp.shape(mu), jnp.shape(sigma))
        self.batch_shape = batch_shape
        self.mu = jnp.broadcast_to(mu, batch_shape)
        self.sigma = jnp.broadcast_to(sigma, batch_shape)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        std_sample = random.normal(rng_key, shape=shape)
        return self.mu + self.sigma * std_sample

    # no need to check on support ]-infty, +infty[
    def logpdf(self, x):
        return scipy.stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)
