import pytest
from jax import numpy as jnp
from jax import random

from mcx.distributions import Poisson


@pytest.fixture
def rng_key():
    return random.PRNGKey(123)


#
# SAMPLING CORRECTNESS
#


#
# LOGPDF CORRECTNESS
#

out_of_support_cases = [
    {"x": -1.0, "expected": -jnp.inf},
    {"x": 1.1, "expected": -jnp.inf},
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = Poisson(1.0).logpdf(case["x"])
    assert logprob == case["expected"]


#
# SAMPLING SHAPE
#


@pytest.mark.parametrize(
    ["lmbda", "sample_shape", "expected_shape"],
    [
        [jnp.array(1), (5,), (5,)],  # 5 1d samples
        [jnp.array([1, 2]), (5,), (5, 2)],  # 5 samples from 2 poisson distributions
        [jnp.array([1, 2]), (5, 2), (5, 2, 2)],
        [
            jnp.array([1, 2, 0, 0]),
            (10,),
            (10, 4),
        ],  # 10 samples from 4 poisson distributions
        [
            jnp.array([[1, 2], [5, 10]]),
            (5, 2),
            (5, 2, 2, 2),
        ],  # 10 samples from a 2x2 batch of Poissons.
    ],
)
def test_sampling_shape(lmbda, expected_shape, sample_shape, rng_key):
    assert Poisson(lmbda=lmbda).sample(rng_key, sample_shape).shape == expected_shape


@pytest.mark.parametrize(
    ["lmbda", "expected_shape"],
    [
        [jnp.array(1), ()],  # 1 sample from 1d poisson
        [jnp.array([1, 2]), (2,)],  # 1 sample from 2 independent poissons
        [jnp.array([[1, 2], [5, 10]]), (2, 2)],  # 2 samples from 2 poissons
    ],
)
def test_sampling_noshape(lmbda, expected_shape, rng_key):
    assert Poisson(lmbda=lmbda).sample(rng_key).shape == expected_shape


#
# LOGPDF SHAPES
#
