from jax import lax
from jax import numpy as jnp
from jax import random
from jax.scipy import stats

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class Pareto(Distribution):
    parameters = {
        "shape": constraints.strictly_positive,
        "scale": constraints.strictly_positive,
    }

    def __init__(self, shape, scale, loc=0):
        self.support = constraints.closed_interval(scale, jnp.inf)
        self.event_shape = ()
        shape, scale, loc = promote_shapes(shape, scale, loc)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(shape), jnp.shape(scale), jnp.shape(loc)
        )
        self.batch_shape = batch_shape
        self.shape = jnp.broadcast_to(shape, batch_shape)
        self.scale = jnp.broadcast_to(scale, batch_shape)
        self.loc = jnp.broadcast_to(loc, batch_shape)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return self.scale * (random.pareto(key=rng_key, b=self.shape, shape=shape))

    @constraints.limit_to_support
    def logpdf(self, x):
        return stats.pareto.logpdf(x=x, b=self.shape, loc=self.loc, scale=self.scale)
