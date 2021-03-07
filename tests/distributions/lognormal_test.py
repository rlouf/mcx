import pytest
from jax import numpy as jnp

from mcx.distributions import LogNormal

#
# SAMPLING CORRECTNESS
#


#
# LOGPDF CORRECTNESS
#

out_of_support_cases = [{"x": 0, "expected": -jnp.inf}, {"x": -1, "expected": -jnp.inf}]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_out_of_support(case):
    logprob = LogNormal(0, 1).logpdf(case["x"])
    assert logprob == case["expected"]


#
# SAMPLING SHAPE
#


#
# LOGPDF SHAPES
#
