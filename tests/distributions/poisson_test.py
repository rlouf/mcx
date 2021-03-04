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

@pytest.mark.parametrize(["lmbda", "sample_shape", "expected_shape"], [
    [jnp.array(1), (5,), (5,)],
    [jnp.array([1,2]), (5,), (5,2)],
    [jnp.array([1,2,0,0]), (10,), (10, 4)],
    [jnp.array([1,2,0,0]), (10,2), (10,2,4)],
])
def test_sampling_shape(lmbda, expected_shape, sample_shape, rng_key):
    assert Poisson(lmbda=lmbda).sample(rng_key, sample_shape).shape == expected_shape


def test_sampling_noshape(rng_key):
    lmbda = jnp.array(1)

    assert Poisson(lmbda=lmbda).sample(rng_key).shape == ()
#
# LOGPDF SHAPES
#
