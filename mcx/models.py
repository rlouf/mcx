import jax.numpy as np

import mcx
import mcx.distributions as mcx


# flake8: noqa
@mcx.model
def neals_funnel(num_dims=10):
    y < ~dist.Normal(0, 3)
    mu = np.zeros(num_dims - 1)
    sigma = np.exp(y / 2)
    x < ~Normal(mu, sigma)
    return x
