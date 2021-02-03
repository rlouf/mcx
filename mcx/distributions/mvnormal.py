from jax import numpy as jnp
from jax import random, scipy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import broadcast_batch_shape


class MvNormal(Distribution):
    params_constraints = {
        "mu": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
    }
    support = constraints.real_vector

    def __init__(self, mu, covariance_matrix):

        (mu_event_shape,) = jnp.shape(mu)[-1:]
        covariance_event_shape = jnp.shape(covariance_matrix)[-2:]
        if (mu_event_shape, mu_event_shape) != covariance_event_shape:
            raise ValueError(
                (
                    f"The number of dimensions implied by `mu` ({mu_event_shape}),"
                    "does not match the dimensions implied by `covariance_matrix` "
                    f"({covariance_event_shape})"
                )
            )

        self.batch_shape = broadcast_batch_shape(
            jnp.shape(mu)[:-1], jnp.shape(covariance_matrix)[:-2]
        )
        self.event_shape = broadcast_batch_shape(
            jnp.shape(mu)[-1:], jnp.shape(covariance_matrix)[-2:]
        )
        self.mu = mu
        self.covariance_matrix = covariance_matrix
        super().__init__()

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        draws = random.multivariate_normal(
            rng_key, mean=self.mu, cov=self.covariance_matrix, shape=shape
        )

        return draws

    # no need to check on support ]-infty, +infty[
    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(
            x, mean=self.mu, cov=self.covariance_matrix
        )
