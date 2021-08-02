import time

import jax.numpy as np
import mcx
import mcx.distributions as dist
import numpy as onp
from jax import random

from sbc_test import run_sbc_test


@mcx.model
def linear_regression(x, lmbda=1.):
    sigma <~ dist.Exponential(lmbda)
    coeffs_init = np.ones(x.shape[-1])
    coeffs <~ dist.Normal(coeffs_init, sigma)
    y = np.dot(x, coeffs)
    predictions <~ dist.Normal(y, sigma)
    return predictions

kernel = mcx.HMC(100)

# ===========

L = 10
num_replicas = 1000
list_rs_coeffs, list_rs_sigma = [], []
rng_key = random.PRNGKey(0)
x_data = np.linspace(0, 5, 1000).reshape(1000, 1)

starttime = time.time()
rs_dict = run_sbc_test(rng_key, linear_regression, kernel, x_data, L, num_replicas)
endtime = time.time()
print(f"\nSBC Running time: {(endtime-starttime)/60:.1f} min")
print(rs_dict)

onp.savetxt("rs_list_coeff_1K.txt", rs_dict['coeffs'])
onp.savetxt("rs_list_sigma_1K.txt", rs_dict['sigma'])
