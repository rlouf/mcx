import pytest
from jax import numpy as jnp
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
    avg = jnp.mean(samples, axis=0).item()
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
    var = jnp.var(samples, axis=0).item()
    assert var == pytest.approx(case["expected"], abs=1e-2)


# # #
# # # LOGPDF CORRECTNESS
# # #

out_of_support_cases = [
    {"a": 1, "b": 1, "n": 10, "x": 11, "expected": -jnp.inf},  # > 0
    {"a": 1, "b": 1, "n": 10, "x": -1, "expected": -jnp.inf},  # < 0
    {"a": 1, "b": 1, "n": 10, "x": 1.1, "expected": -jnp.inf},  # is integer
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = BetaBinomial(n=case["n"], a=case["a"], b=case["b"]).logpdf(case["x"])
    assert logprob == case["expected"]


#
# LOGPDF SHAPES
#

expected_logpdf_shapes = [
    {"a": 1, "b": 1, "n": 10, "x": 1, "expected_shape": ()},
    {"a": jnp.array([1, 2]), "b": jnp.array([1, 2]), "n": 10, "x":1, "expected_shape": (2,)},
]


@pytest.mark.parametrize("case", expected_logpdf_shapes)
def test_logpdf_shape(case):
    logprob = BetaBinomial(n=case["n"], a=case["a"], b=case["b"]).logpdf(case["x"])
    assert logprob.shape == case["expected_shape"]


#
# SAMPLING SHAPE
#

expected_sample_shapes = [
    {"sample_shape": (), "expected_shape": (1,)},
    {"sample_shape": (100,), "expected_shape": (100, 1)},
    {"sample_shape": (100, 10), "expected_shape": (100, 10, 1)},
    {"sample_shape": (1, 100), "expected_shape": (1, 100, 1)},
]


@pytest.mark.parametrize("case", expected_sample_shapes)
def test_sample_shape_scalar_arguments(rng_key, case):
    """Test the correctness of broadcasting when all three arguments are
    scalars. We test scalars arguments separately from array arguments
    since scalars are edge cases when it comes to broadcasting.

    The trailing `1` in the result shapes stands for the batch size.
    """
    samples = BetaBinomial(n=1, a=1, b=1).sample(rng_key, sample_shape=case["sample_shape"])
    assert samples.shape == case["expected_shape"]


expected_sample_shapes_null = [
    {
        "n": 10,
        "a": 1,
        "b": jnp.array([1, 2, 3]),
        "sample_shape": (),
        "expected_shape": (3,),
    },
    {
        "n": 10,
        "a": jnp.array([1, 2, 3]),
        "b": 1,
        "sample_shape": (),
        "expected_shape": (3,),
    },
    {
        "n": jnp.array([10, 20, 30]),
        "a": 2,
        "b": 1,
        "sample_shape": (),
        "expected_shape": (3,),
    },
    {
        "n": 10,
        "a": 1,
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
    {
        "n": 10,
        "a": jnp.array([1, 2]),
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
    {
        "n": jnp.array([[10, 10], [20, 20]]),
        "a": jnp.array([1, 2]),
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
]


@pytest.mark.parametrize("case", expected_sample_shapes_null)
def test_sample_shape_array_arguments_no_sample_shape(rng_key, case):
    """Test the correctness of broadcasting when arguments can be arrays."""
    samples = BetaBinomial(case["n"], case["a"], case["b"]).sample(
        rng_key, case["sample_shape"]
    )
    assert samples.shape == case["expected_shape"]


expected_sample_shapes_one_dim = [
    {
        "n": 10,
        "a": 1,
        "b": jnp.array([1, 2, 3]),
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "n": 10,
        "a": jnp.array([1, 2, 3]),
        "b": 1,
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "n": jnp.array([10, 20, 30]),
        "a": 1,
        "b": 1,
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "n": 10,
        "a": 1,
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
    {
        "n": 10,
        "a": jnp.array([1, 2]),
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
    {
        "n": jnp.array([10, 20]),
        "a": jnp.array([1, 2]),
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
]


@pytest.mark.parametrize("case", expected_sample_shapes_one_dim)
def test_sample_shape_array_arguments_1d_sample_shape(rng_key, case):
    samples = BetaBinomial(case["n"], case["a"], case["b"]).sample(
        rng_key, case["sample_shape"]
    )
    assert samples.shape == case["expected_shape"]


expected_sample_shapes_two_dims = [
    {
        "n": 10,
        "a": 1,
        "b": jnp.array([1, 2, 3]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 3),
    },
    {
        "n": 10,
        "a": jnp.array([1, 2, 3]),
        "b": 1,
        "sample_shape": (100, 3),
        "expected_shape": (100, 3, 3),
    },
    {
        "n": jnp.array([10, 20, 30]),
        "a": 1,
        "b": 1,
        "sample_shape": (100, 3),
        "expected_shape": (100, 3, 3),
    },
    {
        "n": 10,
        "a": 1,
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
    {
        "n": jnp.array([10, 20]),
        "a": jnp.array([1, 2]),
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
]


@pytest.mark.parametrize("case", expected_sample_shapes_two_dims)
def test_sample_shape_array_arguments_2d_sample_shape(rng_key, case):
    samples = BetaBinomial(case["n"], case["a"], case["b"]).sample(
        rng_key, case["sample_shape"]
    )
    assert samples.shape == case["expected_shape"]
