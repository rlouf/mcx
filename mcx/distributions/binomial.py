from functools import partial

from jax import jit
from jax import numpy as np
from jax import random
from jax.scipy.special import gammaln, xlog1py, xlogy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class Binomial(Distribution):
    parameters = {
        "p": constraints.probability,
        "n": constraints.positive_integer,
    }

    def __init__(self, p, n):
        self.support = constraints.integer_interval(0, n)

        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(np.shape(p), np.shape(n))
        self.n = n
        self.p = p

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        n_max = np.max(self.n).item()
        return _random_binomial(rng_key, self.p, self.n, n_max, shape)

    @constraints.limit_to_support
    def logpdf(self, k):
        k = np.floor(k)
        unnormalized = xlogy(k, self.p) + xlog1py(self.n - k, -self.p)
        binomialcoeffln = gammaln(self.n + 1) - (
            gammaln(k + 1) + gammaln(self.n - k + 1)
        )
        return unnormalized + binomialcoeffln


@partial(jit, static_argnums=(3, 4))
def _random_binomial(rng_key, p, n, n_max, shape):
    """Sample from the binomial distribution.

    The general idea is to take `n` samples from a Bernoulli distribution of
    respective parameter `p` and sum the values of these samples.

    However, this is not possible when the distribution is initialized with
    different values of `n`. We thus compute the largest value of `n`, `n_max`
    and take `n_max` samples from the corresponding Bernoulli distributions.

    We then sum the obtained the samples after applying a mask that removes the
    extra samples.
    """
    shape = shape + (n_max,)

    p = np.expand_dims(p, axis=-1)
    n = np.expand_dims(n, axis=-1)

    samples = random.bernoulli(rng_key, p, shape=shape)
    mask = (np.arange(n_max) < n).astype(samples.dtype)
    samples = np.sum(samples * mask, axis=-1)

    return samples
