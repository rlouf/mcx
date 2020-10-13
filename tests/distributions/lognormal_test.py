import pytest
from jax import numpy as np

from mcx.distributions import LogNormal

#
# SAMPLING CORRECTNESS
#


#
# LOGPDF CORRECTNESS
#

out_of_support_cases = [{"x": 0, "expected": -np.inf}, {"x": -1, "expected": -np.inf}]


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
