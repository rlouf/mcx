import numpy as np
import pytest
from jax import numpy as jnp
from jax import random

from mcx.distributions import Gamma


@pytest.fixture
def rng_key():
    return random.PRNGKey(123)


#
# SAMPLING CORRECTNESS
#


def test_sampling_raises_error():
    # Test of jnp.random.gamma implementation used for sampling.
    # Only accepts 'a', not loc and scale
    with pytest.raises(
        TypeError,
        match=r"gamma\(\) takes from 2 to 4 positional arguments but 5 were given",
    ):
        random.gamma(rng_key, jnp.array([1]), jnp.array([1]), jnp.array([2]), ())


cases = [{"loc": 0.0}, {"loc": 10}, {"loc": -5}]


@pytest.mark.parametrize("case", cases)
def test_sampling(case, rng_key):
    dist = Gamma(a=1.0, scale=1.0, **case)
    samples = dist.sample(rng_key, sample_shape=(10000,))
    min_val = np.array(jnp.min(samples))
    np.testing.assert_almost_equal(min_val, case["loc"], decimal=5)


#
# LOGPDF CORRECTNESS
#

out_of_support_cases = [
    {"x": -1.0, "expected": -jnp.inf, "a": 1.0, "loc": 0.0, "scale": 1.0}
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = Gamma(case["a"], case["loc"], case["scale"]).logpdf(case["x"])
    assert logprob == case["expected"]


#
# LOGPDF SHAPES
#

expected_logpdf_shapes = [
    {
        "x": jnp.array([1]),
        "a": jnp.array([0]),
        "loc": jnp.array([0]),
        "scale": jnp.array([1]),
        "expected_shape": (1,),
    },
    {
        "x": jnp.array(1),
        "a": jnp.array(0),
        "loc": jnp.array(0),
        "scale": jnp.array(1),
        "expected_shape": (),
    },
    {
        "x": jnp.ones((5)),
        "a": jnp.array(0),
        "loc": jnp.array(0),
        "scale": jnp.array(1),
        "expected_shape": (5,),
    },
    {
        "x": jnp.ones((8, 1)),
        "a": jnp.array([1, 1]),
        "loc": jnp.array([0, 10]),
        "scale": jnp.array([2, 3]),
        "expected_shape": (8, 2),
    },
    {
        "x": jnp.array([1, 2, 3, 4]).reshape(4, 1),
        "a": jnp.array([1, 4, 10]),
        "loc": jnp.array([0, 1, 1]),
        "scale": jnp.array([3, 2, 1]),
        "expected_shape": (4, 3),
    },
    {
        "x": jnp.array(1),
        "a": jnp.array([1, 2]),
        "loc": jnp.array([0, 1]),
        "scale": jnp.array([5]),
        "expected_shape": (2,),
    },
]


@pytest.mark.parametrize("case", expected_logpdf_shapes)
def test_logpdf_shape(case):
    log_prob = Gamma(a=case["a"], loc=case["loc"], scale=case["scale"]).logpdf(
        case["x"]
    )
    assert log_prob.shape == case["expected_shape"]


#
# SAMPLING SHAPES
#


@pytest.mark.parametrize(
    ["a", "scale", "loc", "sample_shape", "expected_shape"],
    [
        [jnp.array(1), jnp.array(1), jnp.array(1), (5,), (5,)],  # 5 1d samples
        [
            jnp.array([1, 2]),
            jnp.array([1, 1.5]),
            jnp.array([2, 3]),
            (5,),
            (5, 2),
        ],  # 5 samples from 2 gamma distributions
        [jnp.array([1, 2]), jnp.array([1, 2]), jnp.array([1, 2]), (5, 2), (5, 2, 2)],
        [
            jnp.array([1, 2, 3, 4]),
            jnp.array([1, 2, 5, 10]),
            jnp.array([-1, 0, 1, 2]),
            (10,),
            (10, 4),
        ],  # 10 samples from 4 Gamma distributions
        [
            jnp.array([[1, 2], [5, 10]]),
            jnp.array([[1, 2], [4, 6]]),
            jnp.array([[0, -1], [3, 4]]),
            (5, 2),
            (5, 2, 2, 2),
        ],  # 10 samples from a 2x2 batch of Gammas.
    ],
)
def test_sampling_shape(a, scale, loc, sample_shape, expected_shape, rng_key):
    assert (
        Gamma(a=a, scale=scale, loc=loc).sample(rng_key, sample_shape).shape
        == expected_shape
    )


@pytest.mark.parametrize(
    ["a", "scale", "loc", "expected_shape"],
    [
        [jnp.array(1), jnp.array(1), jnp.array(1), ()],  # 1 sample from 1d Gamma
        [
            jnp.array([1, 2]),
            jnp.array([1, 2]),
            jnp.array([1, 2]),
            (2,),
        ],  # 1 sample from 2 independent Gamma
        [
            jnp.array(1),
            jnp.array([1, 2]),
            jnp.array(1),
            (2,),
        ],  # 1 sample from 2 independent Gamma (2d loc)
        [
            jnp.array([5, 5]),
            jnp.array(1),
            jnp.array(1),
            (2,),
        ],  # 1 sample from 2 independent Gamma (2d a)
        [
            jnp.array([[1, 2], [5, 10]]),
            jnp.array([[1, 2], [5, 10]]),
            jnp.array([[1, 2], [5, 10]]),
            (2, 2),
        ],  # 2 samples from 2 Gamma
        [2 * jnp.ones((2, 2, 2)), jnp.array([1]), jnp.array([1]), (2, 2, 2)],
    ],
)
def test_sampling_noshape(a, scale, loc, expected_shape, rng_key):
    assert Gamma(a=a, scale=scale, loc=loc).sample(rng_key).shape == expected_shape
