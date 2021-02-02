import pytest
from jax import numpy as jnp
from jax import random

from mcx.distributions import MvNormal


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


#
# LOGPDF SHAPES
#


expected_logpdf_shapes = [
    {
        "x": jnp.array([1, 2]),
        "mu": jnp.array([0, 1]),
        "covariance_matrix": jnp.array([[1.0, 0.2], [0.2, 1.0]]),
        "expected_shape": (),
    },
    {
        "x": jnp.array([1, 2]),
        "mu": jnp.array([[1, 2], [2, 1]]),
        "covariance_matrix": jnp.array([[1.0, 0.2], [0.2, 1.0]]),
        "expected_shape": (2,),
    },
    {
        "x": jnp.array([1, 2]),
        "mu": jnp.array([[1, 2], [2, 1]]),
        "covariance_matrix": jnp.array(
            [[[1.0, 0.2], [0.2, 1.0]], [[1.0, 0.3], [0.3, 1.0]]]
        ),
        "expected_shape": (2,),
    },
    {
        "x": jnp.array([[1, 2], [2, 3]]),
        "mu": jnp.array([[1, 2], [2, 1]]),
        "covariance_matrix": jnp.array(
            [[[1.0, 0.2], [0.2, 1.0]], [[1.0, 0.3], [0.3, 1.0]]]
        ),
        "expected_shape": (2,),
    },
]


@pytest.mark.parametrize("case", expected_logpdf_shapes)
def test_logpdf_shape(case):
    log_prob = MvNormal(case["mu"], case["covariance_matrix"]).logpdf(case["x"])
    assert log_prob.shape == case["expected_shape"]


#
# SAMPLING SHAPES
#

array2d_argument_expected_shapes = [
    {"sample_shape": (), "expected_shape": (1, 2)},
    {"sample_shape": (100,), "expected_shape": (100, 1, 2)},
    {"sample_shape": (100, 10), "expected_shape": (100, 10, 1, 2)},
    {"sample_shape": (1, 100), "expected_shape": (1, 100, 1, 2)},
]


@pytest.mark.parametrize("case", array2d_argument_expected_shapes)
def test_sample_shape_2darray_argumentse(rng_key, case):
    """Test the correctness of broadcasting when dimension of the MVN is
    two. This is the simplest case of a 2d MVN"""

    samples = MvNormal(
        jnp.array([1.0, 2.0]), jnp.array([[1.0, 0.2], [0.2, 1.0]])
    ).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_zero_dim = [
    {
        "mu": jnp.array([[1.0, 2.0], [2.0, 1.0]]),
        "covariance_matrix": jnp.array([[1, 0.3], [0.3, 1]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
    {
        "mu": jnp.array([1.0, 2.0]),
        "covariance_matrix": jnp.array([[[1, 0.3], [0.3, 1]], [[1, 0.2], [0.2, 1]]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
    {
        "mu": jnp.array([[1.0, 2.0], [2.0, 1.0]]),
        "covariance_matrix": jnp.array([[[1, 0.3], [0.3, 1]], [[1, 0.2], [0.2, 1]]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_zero_dim)
def test_sample_shape_array_arguments_no_sample_shape(rng_key, case):
    """Test the correctness of broadcasting when arguments can be arrays."""
    samples = MvNormal(case["mu"], case["covariance_matrix"]).sample(
        rng_key, case["sample_shape"]
    )
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_one_dim = [
    {
        "mu": jnp.array([[1.0, 2.0], [2.0, 1.0]]),
        "covariance_matrix": jnp.array([[1, 0.3], [0.3, 1]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
    {
        "mu": jnp.array([1.0, 2.0]),
        "covariance_matrix": jnp.array([[[1, 0.3], [0.3, 1]], [[1, 0.2], [0.2, 1]]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
    {
        "mu": jnp.array([[1.0, 2.0], [2.0, 1.0]]),
        "covariance_matrix": jnp.array([[[1, 0.3], [0.3, 1]], [[1, 0.2], [0.2, 1]]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_one_dim)
def test_sample_shape_array_arguments_1d_sample_shape(rng_key, case):
    """Test the correctness of broadcasting when arguments can be arrays
    and sample size is not empty"""
    samples = MvNormal(case["mu"], case["covariance_matrix"]).sample(
        rng_key, case["sample_shape"]
    )
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_two_dim = [
    {
        "mu": jnp.array([[1.0, 2.0], [2.0, 1.0]]),
        "covariance_matrix": jnp.array([[1, 0.3], [0.3, 1]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
    {
        "mu": jnp.array([1.0, 2.0]),
        "covariance_matrix": jnp.array([[[1, 0.3], [0.3, 1]], [[1, 0.2], [0.2, 1]]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
    {
        "mu": jnp.array([[1.0, 2.0], [2.0, 1.0]]),
        "covariance_matrix": jnp.array([[[1, 0.3], [0.3, 1]], [[1, 0.2], [0.2, 1]]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_two_dim)
def test_sample_shape_array_arguments_2d_sample_shape(rng_key, case):
    """Test the correctness of broadcasting when arguments can be arrays
    and sample size is not empty"""
    samples = MvNormal(case["mu"], case["covariance_matrix"]).sample(
        rng_key, case["sample_shape"]
    )
    assert samples.shape == case["expected_shape"]
