from jax import numpy as np
from jax.scipy.special import xlog1py, xlogy, gammaln

from . import constraints
from .bernoulli import Bernoulli
from .distribution import Distribution
from .utils import broadcast_batch_shape


class Binomial(Distribution):
    params_constraints = {"p": constraints.probability}

    def __init__(self, p, n):
        self.support = constraints.integer_interval(0, n)

        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(p, n)
        self.n = n
        self.p = p

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return _random_binomial(rng_key, self.p, self.n, shape)

    def logpdf(self, k):
        unnormalized = xlogy(k, self.p) + xlog1py(self.n - k, -self.p)
        normalization = gammaln(self.n) - gammaln(k) * gammaln(self.n - k)
        return unnormalized / normalization


def _random_binomial(rng_key, p, n, shape):
    """Sample from the binomial distribution.

    The general idea is to take `n` samples from a Bernoulli distribution of
    respective parameter `p` and sum the values of these samples.

    However, this is not possible when the distribution is initialized with
    different values of `n`. We thus compute the largest value of `n`, `n_max`
    and take `n_max` samples from the corresponding Bernoulli distributions.

    We then sum the obtained the samples after applying a mask that removes the
    extra samples.
    """
    n_max = np.max(n)
    bernoulli_sampling_shape = shape + (n_max,)
    samples = Bernoulli(p).sample(rng_key, sample_shape=bernoulli_sampling_shape)

    batch_shape = broadcast_batch_shape(p, n)
    augmented_n = np.ones(shape=batch_shape) * n
    mask = np.arange(n_max) < np.expand_dims(augmented_n, axis=-1)
    samples = np.sum(samples * mask, axis=-1)
    return samples
