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
    coeffs <~ dist.Normal(jnp.zeros(x.shape[-1]), 1)
    predictions <~ dist.Normal(x * coeffs, sigma)
    return predictions

@mcx.model
def linear_regression_mvn(x, lmbda=1.):
    sigma <~ dist.Exponential(lmbda)
    sigma2 <~ dist.Exponential(lmbda)
    rho <~ dist.Uniform(0, 1)
    cov = jnp.array([[sigma, rho*sigma*sigma2],[rho*sigma*sigma2, sigma2]])
    coeffs <~ dist.MvNormal(jnp.ones(x.shape[-1]), cov)
    y = jnp.dot(x, coeffs)
    predictions <~ dist.Normal(y, sigma)
    return predictions
# fmt: on


@pytest.mark.slow
def test_linear_regression():
    x_data = np.random.normal(0, 5, size=1000).reshape(-1, 1)
    y_data = 3 * x_data + np.random.normal(size=x_data.shape)

    kernel = HMC(
        step_size=0.001,
        num_integration_steps=90,
        inverse_mass_matrix=jnp.array([1.0, 1.0]),
    )

    rng_key = jax.random.PRNGKey(2)

    # Batch sampler
    sampler = mcx.sampler(
        rng_key,
        linear_regression,
        (x_data,),
        {"predictions": y_data},
        kernel,
        num_chains=2,
    )
    trace = sampler.run(num_samples=3000)

    mean_coeffs = np.asarray(jnp.mean(trace.raw.samples["coeffs"][:, 1000:], axis=1))
    mean_scale = np.asarray(jnp.mean(trace.raw.samples["sigma"][:, 1000:], axis=1))
    assert mean_coeffs == pytest.approx(3, 1e-1)
    assert mean_scale == pytest.approx(1, 1e-1)


@pytest.mark.slow
def test_linear_regression_mvn():
    # We only check that we can sample, but the results are not checked.
    x_data = np.random.multivariate_normal([0, 1], [[1.0, 0.4], [0.4, 1.0]], size=1000)
    y_data = x_data @ np.array([3, 1]) + np.random.normal(size=x_data.shape[0])

    kernel = HMC(
        num_integration_steps=90,
    )

    rng_key = jax.random.PRNGKey(2)

    # Batch sampler
    sampler = mcx.sampler(
        rng_key,
        linear_regression_mvn,
        (x_data,),
        {"predictions": y_data},
        kernel,
        num_chains=2,
    )
    trace = sampler.run(num_samples=3000)
