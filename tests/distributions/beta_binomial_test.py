import pytest
from jax import numpy as np
from jax import random
from mcx.distributions import BetaBinomial


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


def beta_binomial_mean(n, a, b):
    return (n * a) / (a + b)


def beta_binomial_variance(n, a, b):
    numerator = (n * a * b) * (a + b + n)
    denominator = ((a + b) ** 2) * (a + b + 1)
    return numerator / denominator


#
# SAMPLING CORRECTNESS
#

sample_means = [
    {"n": 1, "a": 0.1, "b": 1, "expected": beta_binomial_mean(n=1, a=0.1, b=1)},
    {"n": 10, "a": 1, "b": 1, "expected": beta_binomial_mean(n=10, a=1, b=1)},
    {"n": 10, "a": 10, "b": 10, "expected": beta_binomial_mean(n=10, a=10, b=10)},
    {"n": 100, "a": 10, "b": 100, "expected": beta_binomial_mean(n=100, a=10, b=100)},
]


@pytest.mark.parametrize("case", sample_means)
def test_sample_mean(rng_key, case):
    samples = BetaBinomial(n=case["n"], a=case["a"], b=case["b"]).sample(
        rng_key, (1_000_000,)
    )
    avg = np.mean(samples, axis=0).item()
    assert avg == pytest.approx(case["expected"], abs=1e-2)


sample_variances = [
    {"n": 10, "a": 1, "b": 1, "expected": beta_binomial_variance(n=10, a=1, b=1)},
    {"n": 1, "a": 0.1, "b": 1, "expected": beta_binomial_variance(n=1, a=0.1, b=1)},
    {"n": 10, "a": 10, "b": 10, "expected": beta_binomial_variance(n=10, a=10, b=10)},
    {"n": 20, "a": 10, "b": 100, "expected": beta_binomial_variance(n=20, a=10, b=100)},
]


@pytest.mark.parametrize("case", sample_variances)
def test_sample_variance(rng_key, case):
    samples = BetaBinomial(n=case["n"], a=case["a"], b=case["b"]).sample(
        rng_key, (1_000_000,)
    )
    var = np.var(samples, axis=0).item()
    assert var == pytest.approx(case["expected"], abs=1e-2)


# #
# # LOGPDF CORRECTNESS
# #

out_of_support_cases = [
    {"a": 1, "b": 1, "n": 10, "x": 11, "expected": -np.inf},  # > 0
    {"a": 1, "b": 1, "n": 10, "x": -1, "expected": -np.inf},  # < 0
    {"a": 1, "b": 1, "n": 10, "x": 1.1, "expected": -np.inf},  # is integer
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = BetaBinomial(n=case["n"], a=case["a"], b=case["b"]).logpdf(case["x"])
    assert logprob == case["expected"]
