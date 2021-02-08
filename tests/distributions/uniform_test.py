import pytest
from jax import numpy as jnp

from mcx.distributions import Uniform

#
# SAMPLING CORRECTNESS
#


#
# LOGPDF CORRECTNESS
#

out_of_support_cases = [
    {"x": 0, "expected": 0.0},  # boundary belongs to support
    {"x": 1, "expected": 0.0},  # boundary belongs to support
    {"x": -0.01, "expected": -jnp.inf},
    {"x": 1.001, "expected": -jnp.inf},
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = Uniform(0, 1).logpdf(case["x"])
    assert logprob == case["expected"]


#
# SAMPLING SHAPE
#


#
# LOGPDF SHAPES
#
