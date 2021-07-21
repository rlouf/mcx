from jax import lax
from jax import numpy as jnp
from jax import random
from jax.scipy.special import gammaln

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class InverseGamma(Distribution):
    parameters = {
        "a": constraints.strictly_positive,
        "b": constraints.strictly_positive,
    }
    support = constraints.strictly_positive

    def __init__(self, a, b):
        self.event_shape = ()
        a, b = promote_shapes(a, b)
        batch_shape = lax.broadcast_shapes(jnp.shape(a), jnp.shape(b))
        self.batch_shape = batch_shape
        self.a = jnp.broadcast_to(a, batch_shape)
        self.b = jnp.broadcast_to(b, batch_shape)

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        # IF X ~ Gamma(a, scale=1/b), then 1/X ~ Inverse-Gamma(a, scale=b)
        return self.b / random.gamma(rng_key, self.a, shape)

    @constraints.limit_to_support
    def logpdf(self, x):
        # We use the fact that f(x;a,b) = f(x/b;a,1) / b to improve
        # numerical stability for small values of ``x`` that can blow up th
        # logp value if not re-scaled.
        y = x / self.b
        return -(self.a + 1) * jnp.log(y) - gammaln(self.a) - 1.0 / y - jnp.log(self.b)
