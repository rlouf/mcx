from jax import numpy as np
import pytest

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
    {"x": -0.01, "expected": -np.inf},
    {"x": 1.001, "expected": -np.inf},
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
