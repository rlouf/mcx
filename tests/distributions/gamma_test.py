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


#
# SAMPLING SHAPES
#
