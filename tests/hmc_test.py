"""Test that the HMC program samples the posterior distribution
correctly.
"""
import jax
import jax.numpy as np
import numpy as onp
import pytest

import mcx
import mcx.distributions as dist
from mcx import HMC


# flake8: noqa: F281
# fmt: off
@mcx.model
def linear_regression(x, lmbda=1.0):
    sigma <~ dist.Exponential(lmbda)
    coeffs_init = np.ones(x.shape[-1])
    coeffs <~ dist.Normal(coeffs_init, sigma)
    y = np.dot(x, coeffs)
    predictions <~ dist.Normal(y, sigma)
    return predictions
# fmt: on


@pytest.mark.sampling
@pytest.mark.slow
def test_linear_regression():

    x_data = onp.random.normal(0, 5, size=1000).reshape(-1, 1)
    y_data = 3 * x_data + onp.random.normal(size=x_data.shape)

    kernel = HMC(
        step_size=0.001,
        num_integration_steps=90,
        inverse_mass_matrix=np.array([1.0, 1.0]),
    )

    observations = {"x": x_data, "predictions": y_data}
    rng_key = jax.random.PRNGKey(2)

    # Batch sampler
    sampler = mcx.sampler(
        rng_key,
        linear_regression,
        kernel,
        num_chains=2,
        **observations,
    )
    trace = sampler.run(num_samples=3000)

    mean_coeffs = onp.asarray(np.mean(trace["posterior"]["coeffs"][:, 1000:], axis=1))
    mean_scale = onp.asarray(np.mean(trace["posterior"]["sigma"][:, 1000:], axis=1))
    assert mean_coeffs == pytest.approx(3, 1e-1)
    assert mean_scale == pytest.approx(1, 1e-1)
