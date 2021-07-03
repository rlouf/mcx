import pytest
from jax import numpy as jnp
from jax import random

from mcx.distributions import HalfNormal


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


#
# LOGPDF SHAPES
#

expected_logpdf_shapes = [
    {"x": 1, "sigma": 1, "expected_shape": ()},
    {"x": 1, "sigma": jnp.array([1, 2]), "expected_shape": (2,)},
]


@pytest.mark.parametrize("case", expected_logpdf_shapes)
def test_logpdf_shape(case):
    log_prob = HalfNormal(case["sigma"]).logpdf(case["x"])
    assert log_prob.shape == case["expected_shape"]


#
# SAMPLING SHAPES
#

scalar_argument_expected_shapes = [
    {"sample_shape": (), "expected_shape": ()},
    {"sample_shape": (100,), "expected_shape": (100,)},
    {
        "sample_shape": (100, 10),
        "expected_shape": (
            100,
            10,
        ),
    },
    {
        "sample_shape": (1, 100),
        "expected_shape": (
            1,
            100,
        ),
    },
]


@pytest.mark.parametrize("case", scalar_argument_expected_shapes)
def test_sample_shape_scalar_arguments(rng_key, case):
    """Test the correctness of broadcasting when both arguments are
    scalars. We test scalars arguments separately from array arguments
    since scalars are edge cases when it comes to broadcasting.

    """
    samples = HalfNormal(1).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_zero_dim = [
    {
        "sigma": jnp.array([1, 2, 3]),
        "sample_shape": (),
        "expected_shape": (3,),
    },
    {
        "sigma": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_zero_dim)
def test_sample_shape_array_arguments_no_sample_shape(rng_key, case):
    """Test the correctness of broadcasting when arguments can be arrays."""
    samples = HalfNormal(case["sigma"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_one_dim = [
    {
        "sigma": jnp.array([1, 2, 3]),
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "sigma": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_one_dim)
def test_sample_shape_array_arguments_1d_sample_shape(rng_key, case):
    samples = HalfNormal(case["sigma"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_two_dims = [
    {
        "sigma": jnp.array([1, 2, 3]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 3),
    },
    {
        "sigma": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_two_dims)
def test_sample_shape_array_arguments_2d_sample_shape(rng_key, case):
    samples = HalfNormal(case["sigma"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


out_of_support_cases = [{"x": -1, "expected": -jnp.inf}]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_out_of_support(case):
    logprob = HalfNormal(1).logpdf(case["x"])
    assert logprob == case["expected"]
