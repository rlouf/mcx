from jax import numpy as np
from jax import random
from scipy.stats import binom as scipy_binom
import pytest

from mcx.distributions import Binomial


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


def binomial_mean(p, n):
    return p * n


def binomial_variance(p, n):
    return n * p * (1.0 - p)


#
# SAMPLING CORRECTNESS
#

sample_means = [
    {"p": 0.1, "n": 100, "expected": binomial_mean(0.1, 100)},
    {"p": 0.5, "n": 1, "expected": binomial_mean(0.5, 1)},
    {"p": 0.9, "n": 10, "expected": binomial_mean(0.9, 10)},
    {"p": 0.9, "n": 100, "expected": binomial_mean(0.9, 100)},
]


@pytest.mark.parametrize("case", sample_means)
def test_sample_mean(rng_key, case):
    samples = Binomial(case["p"], case["n"]).sample(rng_key, (1_000_000,))
    avg = np.mean(samples, axis=0).item()
    assert avg == pytest.approx(case["expected"], abs=1e-2)


sample_variances = [
    {"p": 0.1, "n": 10, "expected": binomial_variance(0.1, 10)},
    {"p": 0.5, "n": 1, "expected": binomial_variance(0.5, 1)},
    {"p": 0.9, "n": 10, "expected": binomial_variance(0.9, 10)},
    {"p": 0.9, "n": 20, "expected": binomial_variance(0.9, 20)},
]


@pytest.mark.parametrize("case", sample_variances)
def test_sample_variance(rng_key, case):
    samples = Binomial(case["p"], case["n"]).sample(rng_key, (1_000_000,))
    var = np.var(samples, axis=0).item()
    assert var == pytest.approx(case["expected"], abs=1e-2)


#
# LOGPDF CORRECTNESS
#

out_of_support_cases = [
    {"p": 0.5, "n": 10, "x": 11, "expected": -np.inf},  # > 0
    {"p": 0.5, "n": 10, "x": -1, "expected": -np.inf},  # < 0
    {"p": 0.5, "n": 10, "x": 1.1, "expected": -np.inf},  # is integer
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = Binomial(case["p"], case["n"]).logpdf(case["x"])
    assert logprob == case["expected"]


edge_cases = [
    {"p": 0, "n": 10, "x": 1, "expected": -np.inf},
    {"p": 1, "n": 10, "x": 1, "expected": -np.inf},
    {"p": 1, "n": 10, "x": 0, "expected": -np.inf},
    {"p": 1.0, "n": 10, "x": 10, "expected": 0.0},
    {"p": 0.0, "n": 10, "x": 0, "expected": 0.0},
]


@pytest.mark.parametrize("case", edge_cases)
def test_logpdf_edge_cases(case):
    """Test following edge cases:

    - The probability of obtaining a non-zero number of successes with a
      probability of success of 0 should be 0;
    - The probability of not obtaining `n` with a probability of success of
      1 should be 0;
    - Conversely, the probability of obtaining 0 successes with a probability
      of success equal to 0 should be 1.
    - And the probability of obtaining `n` successes with a probability of
      success equal to 1 is 1.
    """
    logprob = Binomial(case["p"], case["n"]).logpdf(case["x"])
    assert logprob == case["expected"]


numerical_comparison_cases = [
    {"p": 0.5, "n": 10, "x": 5, "expected": scipy_binom.logpmf(5, 10, 0.5)},
    {"p": 0.9999, "n": 10, "x": 5, "expected": scipy_binom.logpmf(5, 10, 0.9999)},
    {"p": 0.0001, "n": 10, "x": 5, "expected": scipy_binom.logpmf(5, 10, 0.0001)},
]


@pytest.mark.parametrize("case", numerical_comparison_cases)
def test_logpdf_compare_scipy(case):
    logprob = Binomial(case["p"], case["n"]).logpdf(case["x"])
    assert logprob == pytest.approx(case["expected"], abs=1e-2)


#
# LOGPDF SHAPES
#

expected_logpdf_shapes = [
    {"x": 2, "p": 0.5, "n": 10, "expected_shape": ()},
    {"x": 2, "p": np.array([0.1, 0.2]), "n": 10, "expected_shape": (2,)},
]


@pytest.mark.parametrize("case", expected_logpdf_shapes)
def test_logpdf_shape(case):
    log_prob = Binomial(case["p"], case["n"]).logpdf(case["x"])
    assert log_prob.shape == case["expected_shape"]


#
# SAMPLING SHAPE
#

scalar_argument_expected_shapes = [
    {"sample_shape": (), "expected_shape": (1,)},
    {"sample_shape": (100,), "expected_shape": (100, 1)},
    {"sample_shape": (100, 10), "expected_shape": (100, 10, 1)},
    {"sample_shape": (1, 100), "expected_shape": (1, 100, 1)},
]


@pytest.mark.parametrize("case", scalar_argument_expected_shapes)
def test_sample_shape_scalar_arguments(rng_key, case):
    """Test the correctness of broadcasting when both arguments are
    scalars. We test scalars arguments separately from array arguments
    since scalars are edge cases when it comes to broadcasting.

    The trailing `1` in the result shapes stands for the batch size.
    """
    samples = Binomial(0.5, 10).sample(rng_key, case["sample_shape"])
    samples.shape == case["expected_shape"]


array_argument_expected_shapes = [
    {
        "p": 0.5,
        "n": np.array([1, 2, 3, 4]),
        "sample_shape": (),
        "expected_shape": (4,),
    },
    {
        "p": np.array([0.1, 0.2, 0.3]),
        "n": 5,
        "sample_shape": (),
        "expected_shape": (3,),
    },
    {
        "p": 0.5,
        "n": np.array([[1, 2], [3, 4]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
    {
        "p": np.array([0.1, 0.2]),
        "n": np.array([[1, 2], [3, 4]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes)
def test_sample_shape_array_arguments_no_sample_shape(rng_key, case):
    """Test the correctness of broadcasting when arguments can be arrays."""
    samples = Binomial(case["p"], case["n"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_one_dim = [
    {
        "p": 0.1,
        "n": np.array([1, 2, 3]),
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "p": np.array([0.1, 0.2, 0.3]),
        "n": 1,
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "p": 0.1,
        "n": np.array([[1, 2], [3, 4]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
    {
        "p": np.array([0.1, 0.2]),
        "n": np.array([[1, 2], [3, 4]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_one_dim)
def test_sample_shape_array_arguments_1d_sample_shape(rng_key, case):
    samples = Binomial(case["p"], case["n"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_two_dims = [
    {
        "p": 0.1,
        "n": np.array([1, 2, 3]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 3),
    },
    {
        "p": np.array([0.1, 0.2, 0.3]),
        "n": 1,
        "sample_shape": (100, 3),
        "expected_shape": (100, 3, 3),
    },
    {
        "p": 0.1,
        "n": np.array([[1, 2], [3, 4]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
    {
        "p": np.array([0.1, 0.2]),
        "n": np.array([[1, 2], [3, 4]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_two_dims)
def test_sample_shape_array_arguments_2d_sample_shape(rng_key, case):
    samples = Binomial(case["p"], case["n"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]
