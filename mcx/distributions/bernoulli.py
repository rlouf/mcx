from jax import numpy as jnp
from jax import random
from jax.scipy.special import xlog1py, xlogy
from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution


class Bernoulli(Distribution):
    parameters = {"p": constraints.probability}
    support = constraints.boolean

    def __init__(self, p):
        self.event_shape = ()
        self.batch_shape = jnp.shape(p)
        self.p = p * 1.0  # will fail if p is int

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.bernoulli(rng_key, self.p, shape=shape)

    @constraints.limit_to_support
    def logpdf(self, x):
        return xlogy(x, self.p) + xlog1py(1 - x, -self.p)
