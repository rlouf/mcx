import jax
from jax import random

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution


class Dirichlet(Distribution):
    parameters = {"a": constraints.strictly_positive}
    support = constraints.simplex

    def __init__(self, alpha):
        # check that it matches the Categorical's shape conventions
        self.event_shape = alpha.shape[0]
        self.batch_shape = alpha.shape[1]
        self.alpha = alpha

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.dirichlet(rng_key, self.alpha, shape)

    @constraints.limit_to_support
    def logpdf(self, x):
        log_x = jax.nn.log_softmax(x, axis=0)
        unnormalized = jax.numpy.sum((self.alpha - 1) * log_x)
        normalization = jax.numpy.sum(
            jax.numpy.log(jax.scipy.special.lgamma(self.alpha))
        ) - jax.numpy.log(jax.scipy.special.lgamma(jax.numpy.sum(self.alpha)))
        return unnormalized - normalization
