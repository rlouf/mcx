import pytest
from jax import numpy as jnp
from jax import random

from mcx.distributions import Beta


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


def beta_mean(a, b):
    return a / (a + b)


def beta_variance(a, b):
    return (a * b) / ((a + b) ** 2 * (a + b + 1))


#
# SAMPLING CORRECTNESS
#

sample_mean_cases = [
    {"a": 1, "b": 1, "expected": beta_mean(1, 1)},
    {"a": 0.1, "b": 1, "expected": beta_mean(0.1, 1)},
    {"a": 10, "b": 10, "expected": beta_mean(10, 10)},
    {"a": 10, "b": 100, "expected": beta_mean(10, 100)},
]


@pytest.mark.parametrize("case", sample_mean_cases)
def test_sample_mean(rng_key, case):
    samples = Beta(case["a"], case["b"]).sample(rng_key, (100_000,))
    avg = jnp.mean(samples, axis=0).item()
    assert avg == pytest.approx(case["expected"], abs=1e-2)


sample_variance_cases = [
    {"a": 1, "b": 1, "expected": beta_variance(1, 1)},
    {"a": 0.1, "b": 1, "expected": beta_variance(0.1, 1)},
    {"a": 10, "b": 10, "expected": beta_variance(10, 10)},
    {"a": 10, "b": 100, "expected": beta_variance(10, 100)},
]


@pytest.mark.parametrize("case", sample_variance_cases)
def test_sample_variance(rng_key, case):
    samples = Beta(case["a"], case["b"]).sample(rng_key, (100_000,))
    var = jnp.var(samples, axis=0).item()
    assert var == pytest.approx(case["expected"], abs=1e-2)


#
# LOGPDF CORRECTNESS
#   We trust the implementation in `jax.scipy.stats` for numerical correctness.
#

out_of_support_cases = [
    {"x": -0.1, "expected": -jnp.inf},
    {"x": 1.1, "expected": -jnp.inf},
    {
        "x": 0,
        "expected": -jnp.inf,
    },  # the logpdf is defined on the *open* interval ]0, 1[
    {"x": 1, "expected": -jnp.inf},  # idem
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = Beta(1, 1).logpdf(case["x"])
    assert logprob == case["expected"]


#
# LOGPDF SHAPES
#

expected_logpdf_shapes = [
    {"x": 0.5, "a": 0, "b": 1, "expected_shape": ()},
    {"x": 0.5, "a": jnp.array([1, 2]), "b": 1, "expected_shape": (2,)},
]


@pytest.mark.parametrize("case", expected_logpdf_shapes)
def test_logpdf_shape(case):
    log_prob = Beta(case["a"], case["b"]).logpdf(case["x"])
    assert log_prob.shape == case["expected_shape"]


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
    """Test the correctness of broadcasting when both arguments are
    scalars. We test scalars arguments separately from array arguments
    since scalars are edge cases when it comes to broadcasting.

    The trailing `1` in the result shapes stands for the batch size.
    """
    samples = Beta(1, 1).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


expected_sample_shapes_null = [
    {"a": 1, "b": jnp.array([1, 2, 3]), "sample_shape": (), "expected_shape": (3,)},
    {"a": jnp.array([1, 2, 3]), "b": 1, "sample_shape": (), "expected_shape": (3,)},
    {
        "a": 1,
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
    {
        "a": jnp.array([1, 2]),
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
]


@pytest.mark.parametrize("case", expected_sample_shapes_null)
def test_sample_shape_array_arguments_no_sample_shape(rng_key, case):
    """Test the correctness of broadcasting when arguments can be arrays."""
    samples = Beta(case["a"], case["b"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


expected_sample_shapes_one_dim = [
    {
        "a": 1,
        "b": jnp.array([1, 2, 3]),
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "a": jnp.array([1, 2, 3]),
        "b": 1,
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "a": 1,
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
    {
        "a": jnp.array([1, 2]),
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
]


@pytest.mark.parametrize("case", expected_sample_shapes_one_dim)
def test_sample_shape_array_arguments_1d_sample_shape(rng_key, case):
    samples = Beta(case["a"], case["b"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


expected_sample_shapes_two_dims = [
    {
        "a": 1,
        "b": jnp.array([1, 2, 3]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 3),
    },
    {
        "a": jnp.array([1, 2, 3]),
        "b": 1,
        "sample_shape": (100, 3),
        "expected_shape": (100, 3, 3),
    },
    {
        "a": 1,
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
    {
        "a": jnp.array([1, 2]),
        "b": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
]


@pytest.mark.parametrize("case", expected_sample_shapes_two_dims)
def test_sample_shape_array_arguments_2d_sample_shape(rng_key, case):
    samples = Beta(case["a"], case["b"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]
