"""Test that the HMC program samples the posterior distribution
correctly.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mcx
import mcx.distributions as dist
from mcx import HMC


# flake8: noqa: F281
# fmt: off
@mcx.model
def linear_regression(x, lmbda=1.0):
    sigma <~ dist.Exponential(lmbda)
    coeffs_init = jnp.ones(x.shape[-1])
    coeffs <~ dist.Normal(coeffs_init, sigma)
    y = jnp.dot(x, coeffs)
    predictions <~ dist.Normal(y, sigma)
    return predictions
# fmt: on


@pytest.mark.sampling
@pytest.mark.slow
def test_linear_regression():
    x_data = np.random.normal(0, 5, size=1000).reshape(-1, 1)
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)

    kernel = HMC(
        step_size=0.001,
        num_integration_steps=90,
        inverse_mass_matrix=jnp.array([1.0, 1.0]),
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

    mean_coeffs = np.asarray(jnp.mean(trace.raw.samples["coeffs"][:, 1000:], axis=1))
    mean_scale = np.asarray(jnp.mean(trace.raw.samples["sigma"][:, 1000:], axis=1))
    assert mean_coeffs == pytest.approx(3, 1e-1)
    assert mean_scale == pytest.approx(1, 1e-1)
