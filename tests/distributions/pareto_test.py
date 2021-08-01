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
        denominator = ((shape - 1) ** 2) * (shape - 2)
        return numerator / denominator


#
# SAMPLING CORRECTNESS
#

sample_means = [
    {"shape": 2, "scale": 0.1, "expected": pareto_mean(shape=2, scale=0.1)},
    {"shape": 10, "scale": 1, "expected": pareto_mean(shape=10, scale=1)},
    {"shape": 10, "scale": 10, "expected": pareto_mean(shape=10, scale=10)},
    {"shape": 100, "scale": 10, "expected": pareto_mean(shape=100, scale=10)},
]


@pytest.mark.parametrize("case", sample_means)
def test_sample_mean(rng_key, case):
    samples = Pareto(shape=case["shape"], scale=case["scale"]).sample(
        rng_key, (1_000_000,)
    )
    avg = jnp.mean(samples, axis=0).item()
    assert avg == pytest.approx(case["expected"], abs=1e-2)


sample_variances = [
    {"shape": 2, "scale": 0.1, "expected": pareto_variance(shape=2, scale=0.1)},
    {"shape": 10, "scale": 1, "expected": pareto_variance(shape=10, scale=1)},
    {"shape": 10, "scale": 10, "expected": pareto_variance(shape=10, scale=10)},
    {"shape": 100, "scale": 10, "expected": pareto_variance(shape=100, scale=10)},
]


@pytest.mark.parametrize("case", sample_variances)
def test_sample_variance(rng_key, case):
    samples = Pareto(shape=case["shape"], scale=case["scale"]).sample(
        rng_key, (1_000_000,)
    )
    var = jnp.var(samples, axis=0).item()
    assert var == pytest.approx(case["expected"], abs=1e-2)


#
# LOGPDF CORRECTNESS
#

out_of_support_cases = [
    {"shape": 3, "scale": 1, "x": 0.5, "expected": -jnp.inf},
    {"shape": 3, "scale": 1, "x": -1, "expected": -jnp.inf},
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = Pareto(shape=case["shape"], scale=case["scale"]).logpdf(case["x"])
    assert logprob == case["expected"]


#
# LOGPDF SHAPES
#

expected_logpdf_shapes = [
    {"shape": 3, "scale": 1, "x": 1, "expected_shape": ()},
    {"shape": 3, "scale": 1, "x": jnp.array([1, 2]), "expected_shape": (2,)},
]


@pytest.mark.parametrize("case", expected_logpdf_shapes)
def test_logpdf_shape(case):
    log_prob = Pareto(shape=case["shape"], scale=case["scale"]).logpdf(case["x"])
    assert log_prob.shape == case["expected_shape"]


#
# SAMPLING SHAPE
#

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
    samples = Pareto(scale=1, shape=1).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_zero_dim = [
    {
        "shape": 1,
        "scale": jnp.array([1, 2, 3]),
        "sample_shape": (),
        "expected_shape": (3,),
    },
    {
        "shape": jnp.array([1, 2, 3]),
        "scale": 1,
        "sample_shape": (),
        "expected_shape": (3,),
    },
    {
        "shape": 1,
        "scale": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
    {
        "shape": jnp.array([1, 2]),
        "scale": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_zero_dim)
def test_sample_shape_array_arguments_no_sample_shape(rng_key, case):
    """Test the correctness of broadcasting when arguments can be arrays."""
    samples = Pareto(shape=case["shape"], scale=case["scale"]).sample(
        rng_key, case["sample_shape"]
    )
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_one_dim = [
    {
        "shape": 1,
        "scale": jnp.array([1, 2, 3]),
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "shape": jnp.array([1, 2, 3]),
        "scale": 1,
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "shape": 1,
        "scale": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
    {
        "shape": jnp.array([1, 2]),
        "scale": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100,),
        "expected_shape": (100, 2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_one_dim)
def test_sample_shape_array_arguments_1d_sample_shape(rng_key, case):
    samples = Pareto(shape=case["shape"], scale=case["scale"]).sample(
        rng_key, case["sample_shape"]
    )
    assert samples.shape == case["expected_shape"]


array_argument_expected_shapes_two_dims = [
    {
        "shape": 1,
        "scale": jnp.array([1, 2, 3]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 3),
    },
    {
        "shape": jnp.array([1, 2, 3]),
        "scale": 1,
        "sample_shape": (100, 3),
        "expected_shape": (100, 3, 3),
    },
    {
        "shape": 1,
        "scale": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
    {
        "shape": jnp.array([1, 2]),
        "scale": jnp.array([[1, 2], [3, 4]]),
        "sample_shape": (100, 2),
        "expected_shape": (100, 2, 2, 2),
    },
]


@pytest.mark.parametrize("case", array_argument_expected_shapes_two_dims)
def test_sample_shape_array_arguments_2d_sample_shape(rng_key, case):
    samples = Pareto(shape=case["shape"], scale=case["scale"]).sample(
        rng_key, case["sample_shape"]
    )
    assert samples.shape == case["expected_shape"]
