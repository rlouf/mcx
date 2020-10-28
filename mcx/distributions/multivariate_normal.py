from jax import numpy as np

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution


class MultivariateNormal(Distribution):
    params_constraints = {
        "mu": constraints.real_vector,
        "covariance": constraints.symmetric_positive_definite,
    }
    support = constraints.real_vector

    def __init__(self, mu, covariance):
        self.event_shape = np.shape(covariance)[-1]
        self.batch_shape = broadcast_batch_shape(np.shape(mu), np.shape(covariance))
