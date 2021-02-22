from jax import lax
from jax import numpy as jnp
from jax import random

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class Uniform(Distribution):
    parameters = {"lower": constraints.real, "upper": constraints.real}

    def __init__(self, lower, upper):
        self.support = constraints.closed_interval(lower, upper)

        self.event_shape = ()
        lower, upper = promote_shapes(lower, upper)
        batch_shape = lax.broadcast_shapes(jnp.shape(lower), jnp.shape(upper))
        self.batch_shape = batch_shape
        self.lower = jnp.broadcast_to(lower, batch_shape)
        self.upper = jnp.broadcast_to(upper, batch_shape)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        u = random.uniform(rng_key, shape)
        return u * (self.upper - self.lower) + self.lower

    @constraints.limit_to_support
    def logpdf(self, x):
        return jnp.log(1.0 / (self.upper - self.lower))
