import pytest
from jax import numpy as jnp
from jax import random

from mcx.distributions import Pareto


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


def pareto_mean(shape, scale):
    if shape > 1:
        return shape * scale / (shape - 1.0)
    else:
        return jnp.inf


def pareto_variance(shape, scale):
    if shape <= 2:
        return jnp.inf
    else:
        numerator = (scale ** 2) * shape
        denominator = ((shape - 1) **2 ) * (shape - 2)
        return numerator / denominator


#
# SAMPLING CORRECTNESS
#

sample_means = [
    {"shape": 2, "scale": 0.1, "expected": pareto_mean(shape=2, scale=0.1)},
    {"shape": 10, "scale": 1, "expected": pareto_mean(shape=10, scale=1)},
    {"shape": 10, "scale": 10,  "expected": pareto_mean(shape=10, scale=10)},
    {"shape": 100, "scale": 10, "expected": pareto_mean(shape=100, scale=10)},
]


@pytest.mark.parametrize("case", sample_means)
def test_sample_mean(rng_key, case):
    samples = Pareto(shape=case["shape"], scale=case["scale"]).sample(
        rng_key, (1_000_000,)
    )
    avg = jnp.mean(samples, axis=0).item()
    assert avg == pytest.approx(case["expected"], abs=1e-2)
