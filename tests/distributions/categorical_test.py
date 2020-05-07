from jax import numpy as np
from jax import random
import pytest

from mcx.distributions import Categorical


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


#
# LOGPDF CORRECTNESS
#

out_of_support_cases = [
    {"probs": np.array([0.1, 0.2, 0.7]), "x": -1, "expected": -np.inf},
    {"probs": np.array([0.1, 0.2, 0.7]), "x": 3, "expected": -np.inf},
    {"probs": np.array([0.1, 0.2, 0.7]), "x": 3.5, "expected": -np.inf},
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = Categorical(case["probs"]).logpdf(case["x"])
    assert logprob == case["expected"]


edge_cases = [
    {"probs": np.array([1]), "x": 0, "expected": 0.0},
    {"probs": np.array([0, 1]), "x": 1, "expected": 0},
    {"probs": np.array([0, 1]), "x": 0, "expected": -np.inf},
]


@pytest.mark.parametrize("case", edge_cases)
def test_logpdf_edge_cases(case):
    logprob = Categorical(case["probs"]).logpdf(case["x"])
    assert logprob == case["expected"]


#
# SAMPLING SHAPE
#

single_parameter_expected_shapes = [
    {"sample_shape": (), "expected_shape": (1,)},
    {"sample_shape": (100,), "expected_shape": (100, 1)},
    {"sample_shape": (100, 10), "expected_shape": (100, 10, 1)},
    {"sample_shape": (1, 100), "expected_shape": (1, 100, 1)},
]


@pytest.mark.parametrize("case", single_parameter_expected_shapes)
def test_sample_single_parameter(rng_key, case):
    samples = Categorical(np.array([0.2, 0.7, 0.1])).sample(
        rng_key, case["sample_shape"]
    )
    assert samples.shape == case["expected_shape"]


array_of_parameters_expected_shapes = [
    {
        "probs": np.array([[0.1, 0.9], [0.2, 0.3], [0.5, 0.5]]),
        "sample_shape": (),
        "expected_shape": (3,),
    },
    {
        "probs": np.array([[0.1, 0.9], [0.2, 0.3], [0.5, 0.5]]),
        "sample_shape": (100,),
        "expected_shape": (100, 3),
    },
    {
        "probs": np.array([[0.1, 0.9], [0.2, 0.3], [0.5, 0.5]]),
        "sample_shape": (100, 10),
        "expected_shape": (100, 10, 3),
    },
    {
        "probs": np.array(
            [
                [[0.1, 0.9], [0.2, 0.3], [0.5, 0.5]],
                [[0.1, 0.9], [0.2, 0.3], [0.5, 0.5]],
            ]
        ),
        "sample_shape": (100, 10),
        "expected_shape": (100, 10, 2, 3),
    },
]


@pytest.mark.parametrize("case", array_of_parameters_expected_shapes)
def test_sample_array_of_parameters(rng_key, case):
    samples = Categorical(case["probs"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


#
# LOGPDF SHAPES
#

logpdf_shapes = [
    {"x": 0, "probs": np.array([0.4, 0.1, 0.5]), "expected_shape": ()},
    {"x": 1, "probs": np.array([[0.1, 0.9], [0.2, 0.8]]), "expected_shape": (2,)},
]


@pytest.mark.parametrize("case", logpdf_shapes)
def test_logpdf_shape(case):
    log_prob = Categorical(case["probs"]).logpdf(case["x"])
    assert log_prob.shape == case["expected_shape"]
