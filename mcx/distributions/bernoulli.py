from jax import numpy as jnp
from jax import random
from jax.scipy.special import xlog1py, xlogy
from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class Bernoulli(Distribution):
    parameters = {"p": constraints.probability}
    support = constraints.boolean

    def __init__(self, p):
        self.event_shape = ()
        batch_shape, (p,) = promote_shapes(p)
        self.batch_shape = batch_shape
        self.p = jnp.broadcast_to(p * 1.0, batch_shape)  # will fail if p is int

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.bernoulli(rng_key, self.p, shape=shape)

    @constraints.limit_to_support
    def logpdf(self, x):
        return xlogy(x, self.p) + xlog1py(1 - x, -self.p)
