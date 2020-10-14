import pytest
from jax import numpy as np

from mcx.distributions import DiscreteUniform


def discreteuniform_mean(lower, upper):
    return (lower + upper) / 2.0


def discreteuniform_variance(lower, upper):
    return (np.pow((upper - lower + 1.0), 2) - 1.0) / 12.0


#
# SAMPLING CORRECTNESS
#


#
# LOGPDF CORRECTNESS
#

out_of_support_cases = [
    {"lower": 1, "upper": 10, "x": 11, "expected": -np.inf},  # > upper
    {"lower": 0, "upper": 3, "x": -1, "expected": -np.inf},  # < lower
    {"lower": 0, "upper": 4, "x": 1.3, "expected": -np.inf},  # float
]


@pytest.mark.parametrize("case", out_of_support_cases)
def test_logpdf_out_of_support(case):
    logprob = DiscreteUniform(case["lower"], case["upper"]).logpdf(case["x"])
    assert logprob == case["expected"]


edge_cases = [
    {"lower": 1, "upper": 1, "x": 1, "expected": 0},
    {"lower": 0, "upper": 3, "x": 0, "expected": -np.log(4)},
    {"lower": 0, "upper": 3, "x": 3, "expected": -np.log(4)},
]


@pytest.mark.parametrize("case", edge_cases)
def test_logpdf_edge_cases(case):
    logprob = DiscreteUniform(case["lower"], case["upper"]).logpdf(case["x"])
    assert logprob == case["expected"]


#
# SAMPLING SHAPE
#


#
# LOGPDF SHAPES
#
