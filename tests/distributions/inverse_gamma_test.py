import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

from mcx.distributions import InverseGamma


@pytest.fixture
def rng_key():
    return random.PRNGKey(123)


#
# SAMPLING CORRECTNESS
#


def invgamma_mean(a, b):
    # only defined for a > 1
    return b / (a - 1)


def invgamma_variance(a, b):
    # only defined for a > 2
    return (b ** 2) / ((a - 1) ** 2 * (a - 2))


sample_mean_cases = [
    {"a": 5.5, "b": 5, "expected": invgamma_mean(5.5, 5)},
    {"a": 15, "b": 2.0, "expected": invgamma_mean(15, 2.0)},
    {"a": 20, "b": 1.5, "expected": invgamma_mean(20, 1.5)},
]


@pytest.mark.parametrize("case", sample_mean_cases)
def test_sample_mean(rng_key, case):
    samples = InverseGamma(case["a"], case["b"]).sample(rng_key, (100_000,))
    avg = jnp.mean(samples, axis=0).item()
    np.testing.assert_almost_equal(avg, case["expected"], decimal=2)


sample_variance_cases = [
    {"a": 5.5, "b": 5, "expected": invgamma_variance(5.5, 5)},
    {"a": 15, "b": 2.0, "expected": invgamma_variance(15, 2.0)},
    {"a": 20, "b": 1.5, "expected": invgamma_variance(20, 1.5)},
]


@pytest.mark.parametrize("case", sample_variance_cases)
def test_sample_variance(rng_key, case):
    samples = InverseGamma(case["a"], case["b"]).sample(rng_key, (100_000,))
    var = jnp.var(samples, axis=0).item()
    np.testing.assert_almost_equal(var, case["expected"], decimal=2)


#
# LOGPDF SHAPES
#

expected_logpdf_shapes = [
    {
        "x": jnp.array([1]),
        "a": jnp.array([0]),
        "b": jnp.array([1]),
        "expected_shape": (1,),
    },
    {
        "x": jnp.array(1),
        "a": jnp.array(0),
        "b": jnp.array(1),
        "expected_shape": (),
    },
    {
        "x": jnp.ones((5)),
        "a": jnp.array(0),
        "b": jnp.array(1),
        "expected_shape": (5,),
    },
    {
        "x": jnp.ones((8, 1)),
        "a": jnp.array([1, 1]),
        "b": jnp.array([2, 3]),
        "expected_shape": (8, 2),
    },
    {
        "x": jnp.array([1, 2, 3, 4]).reshape(4, 1),
        "a": jnp.array([1, 4, 10]),
        "b": jnp.array([3, 2, 1]),
        "expected_shape": (4, 3),
    },
    {
        "x": jnp.array(1),
        "a": jnp.array([1, 2]),
        "b": jnp.array([5]),
        "expected_shape": (2,),
    },
]


@pytest.mark.parametrize("case", expected_logpdf_shapes)
def test_logpdf_shape(case):
    log_prob = InverseGamma(a=case["a"], b=case["b"]).logpdf(case["x"])
    assert log_prob.shape == case["expected_shape"]


#
# SAMPLING SHAPES
#


@pytest.mark.parametrize(
    ["a", "b", "sample_shape", "expected_shape"],
    [
        # 5 1d samples
        [jnp.array(1), jnp.array(1), (5,), (5,)],
        # 5 samples from 2 inverse-gamma distributions
        [jnp.array([1, 2]), jnp.array([1, 1.5]), (5,), (5, 2)],
        [jnp.array([1, 2]), jnp.array([1, 2]), (5, 2), (5, 2, 2)],
        # 10 samples from 4 inverse-gamma distributions
        [jnp.array([1, 2, 3, 4]), jnp.array([1, 2, 5, 10]), (10,), (10, 4)],
        # 10 samples from a 2x2 batch of inverse-gammas.
        [
            jnp.array([[1, 2], [5, 10]]),
            jnp.array([[1, 2], [4, 6]]),
            (5, 2),
            (5, 2, 2, 2),
        ],
    ],
)
def test_sampling_shape(a, b, sample_shape, expected_shape, rng_key):
    assert InverseGamma(a=a, b=b).sample(rng_key, sample_shape).shape == expected_shape
