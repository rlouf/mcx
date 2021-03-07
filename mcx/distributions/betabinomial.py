from jax import lax
from jax import numpy as jnp
from jax import random
from jax.scipy import stats

from mcx.distributions import constraints
from mcx.distributions.binomial import _random_binomial
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class BetaBinomial(Distribution):
    parameters = {
        "n": constraints.positive_integer,
        "a": constraints.strictly_positive,
        "b": constraints.strictly_positive,
    }

    def __init__(self, n, a, b):
        self.support = constraints.integer_interval(0, n)
        self.event_shape = ()
        n, a, b = promote_shapes(n, a, b)
        batch_shape = lax.broadcast_shapes(jnp.shape(n), jnp.shape(a), jnp.shape(b))
        self.batch_shape = batch_shape
        self.a = jnp.broadcast_to(a, batch_shape)
        self.b = jnp.broadcast_to(b, batch_shape)
        self.n = jnp.broadcast_to(n, batch_shape)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        p = random.beta(rng_key, self.a, self.b, shape=shape)
        n_max = jnp.max(self.n).item()
        return _random_binomial(rng_key, p, self.n, n_max, shape)

    @constraints.limit_to_support
    def logpdf(self, x):
        return stats.betabinom.logpmf(x, self.n, self.a, self.b)
