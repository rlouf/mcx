from jax import lax
from jax import numpy as jnp
from jax import random
from jax.scipy import stats

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class Gamma(Distribution):
    parameters = {
        "a": constraints.strictly_positive,
        "loc": constraints.real,
        "scale": constraints.strictly_positive,
    }
    support = constraints.strictly_positive

    def __init__(self, a, loc, scale):
        self.event_shape = ()
        a, loc, scale = promote_shapes(a, loc, scale)
        batch_shape = lax.broadcast_shapes(
            jnp.shape(a), jnp.shape(loc), jnp.shape(scale)
        )
        self.batch_shape = batch_shape
        self.a = jnp.broadcast_to(a, batch_shape)
        self.loc = jnp.broadcast_to(loc, batch_shape)
        self.scale = jnp.broadcast_to(scale, batch_shape)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.gamma(rng_key, self.a, self.loc, self.scale, shape)

    @constraints.limit_to_support
    def logpdf(self, x):
        return stats.gamma(x, self.a, self.loc, self.scale)
