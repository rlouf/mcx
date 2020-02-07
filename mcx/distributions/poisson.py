from functools import partial

import jax
from jax import lax
from jax import numpy as np
from jax import random
from jax.scipy.special import xlogy

from . import constraints
from .distribution import Distribution
from .utils import broadcast_batch_shape, limit_to_support


class Poisson(Distribution):
    parameters = {"lambda": constraints.positive}
    support = constraints.positive_integer

    def __init__(self, lmbda):
        self.event_shape = ()
        self.batch_shape = broadcast_batch_shape(lmbda)
        self.lmbda = lmbda

    def sample(self, rng_key, sample_shape):
        shape = sample_shape + self.batch_shape + self.event_shape
        return _random_poisson(rng_key, self.lmbda, shape)

    @limit_to_support
    def logpdf(self, x):
        x = x * 1.0
        return lax.add(
            xlogy(x, lax.log(self.lmbda)),
            lax.add(lax.neg(self.lmbda), lax.neg(lax.lgamma(x))),
        )


@partial(jax.jit, static_argnums=(2,))
def _random_poisson(rng_key, lmbda, shape):
    """
    References
    ----------
    .. [1] Knuth, Donald E. Art of computer programming, volume 2:
           Seminumerical algorithms. Addison-Wesley Professional, 2014 (p 137).
    """
    L = lax.exp(lax.neg(lmbda))
    k = np.zeros(shape=shape)
    p = np.ones(shape=shape)

    is_done = p < L
    while not is_done.all():
        _, rng_key = random.split(rng_key)
        u = random.uniform(rng_key, shape=shape)
        p = np.where(is_done, p, u * p)
        k = np.where(is_done, k, k + 1)
        is_done = p < L

    return k
