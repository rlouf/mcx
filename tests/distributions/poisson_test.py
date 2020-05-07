from jax import numpy as np
import pytest

from mcx.distributions import Poisson


#
# SAMPLING CORRECTNESS
#


#
# LOGPDF CORRECTNESS
#

out_of_support_cases = [
    {"x": -1.0, "expected": -np.inf},
    {"x": 1.1, "expected": -np.inf},
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = Poisson(1.0).logpdf(case["x"])
    assert logprob == case["expected"]


#
# SAMPLING SHAPE
#


#
# LOGPDF SHAPES
#
