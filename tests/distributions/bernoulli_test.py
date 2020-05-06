import jax
from jax import numpy as np
import numpy as onp
import pytest

from mcx.distributions import Bernoulli


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)


#
# SAMPLING CORRECTNESS
#

@pytest.mark.parametrize("p", [0, 1, 0.5])
def test_sample_frequency(rng_key, p):
    samples = Bernoulli(p).sample(rng_key, (1_000_000,))
    avg = np.mean(samples, axis=0).item()
    assert avg == pytest.approx(p, abs=1e-3)


def test_sample_frequency_vectorized(rng_key):
    probabilities = np.array([0, 1, 0.5])
    samples = Bernoulli(probabilities).sample(rng_key, (1_000_000,))
    averages = np.mean(samples, axis=0)

    p_array = onp.asarray(probabilities)
    avg_array = onp.asarray(averages)
    assert avg_array == pytest.approx(p_array, abs=1e-3)


#
# LOGPDF CORRECTNESS
#

edge_cases = [
    {"p": 1, "x": 0, "expected": -np.inf},
    {"p": 0, "x": 1, "expected": -np.inf},
]


@pytest.mark.parametrize("case", edge_cases)
def test_logpdf_edge_cases(case):
    logprob = Bernoulli(case["p"]).logpdf(case["x"])
    assert logprob.item() == pytest.approx(case["expected"])


out_of_support_cases = [
    {"p": 0.5, "x": 0.5, "expected": -np.inf},
    {"p": 0.5, "x": 3, "expected": -np.inf},
    {"p": 0.5, "x": -1.1, "expected": -np.inf},
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = Bernoulli(case["p"]).logpdf(case["x"])
    assert logprob.item() == case["expected"]


example_values = [
    {"p": 0.5, "x": 0, "expected": -0.6931471805599453},
    {"p": 0.5, "x": 1, "expected": -0.6931471805599453},
    {"p": 0.2, "x": 1, "expected": -1.6094379124341003},
    {"p": 0.2, "x": 0, "expected": -0.22314355131420976},
    {"p": 0.7, "x": 1, "expected": -0.35667494393873245},
    {"p": 0.7, "x": 0, "expected": -1.203972804325936},
]


@pytest.mark.parametrize("example", example_values)
def test_logpdf_example_values(rng_key, example):
    """The values are obtained from `scipy.stats.bernoulli.logpmf`.
    """
    logprob = Bernoulli(example["p"]).logpdf(example["x"])
    assert logprob.item() == pytest.approx(example["expected"], abs=1e-7)


#
# LOGPDF SHAPES
#

expected_logpdf_shapes = [
    {"x": 1, "p": 0, "expected_shape": ()},
    {"x": 1, "p": np.array([0.1, 0.2]), "expected_shape": (2,)},
]


@pytest.mark.parametrize("case", expected_logpdf_shapes)
def test_logpdf_shape(case):
    log_prob = Bernoulli(case["p"]).logpdf(case["x"])
    assert log_prob.shape == case["expected_shape"]


#
# SAMPLING SHAPES
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
    samples = Bernoulli(0.5).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]


expected_sample_shapes = [
    {"p": np.array([0.3, 0.5]), "sample_shape": (), "expected_shape": (2,)},
    {
        "p": np.array([[0.3, 0.5], [0.1, 0.2]]),
        "sample_shape": (),
        "expected_shape": (2, 2),
    },
    {"p": np.array([0.3, 0.5]), "sample_shape": (10,), "expected_shape": (10, 2)},
    {
        "p": np.array([[0.3, 0.5], [0.1, 0.2]]),
        "sample_shape": (10,),
        "expected_shape": (10, 2, 2),
    },
    {
        "p": np.array([0.3, 0.5]),
        "sample_shape": (10, 10),
        "expected_shape": (10, 10, 2),
    },
    {
        "p": np.array([[0.3, 0.5], [0.1, 0.2]]),
        "sample_shape": (10, 10),
        "expected_shape": (10, 10, 2, 2),
    },
]


@pytest.mark.parametrize("case", expected_sample_shapes)
def test_sample_shape_array_parameter(rng_key, case):
    samples = Bernoulli(case["p"]).sample(rng_key, case["sample_shape"])
    assert samples.shape == case["expected_shape"]
