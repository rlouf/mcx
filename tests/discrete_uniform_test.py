import unittest

from jax import numpy as np
from jax import random

from mcmx.distributions import DiscreteUniform


class DiscreteUniformTest(unittest.TestCase):
    def setUp(self):
        self.rng_key = random.PRNGKey(0)

    #
    # SAMPLING CORRECTNESS
    #

    def test_sample_mean(self):
        pass

    def test_sample_variance(self):
        pass

    #
    # LOGPDF CORRECTNESS
    #

    def test_logpdf_out_of_support(self):
        test_cases = [
            {"lower": 1, "upper": 10, "x": 11, "expected": -np.inf},  # > upper
            {"lower": 0, "upper": 3, "x": -1, "expected": -np.inf},  # < lower
            {"lower": 0, "upper": 4, "x": 1.3, "expected": -np.inf},  # float
        ]
        for case in test_cases:
            logprob = DiscreteUniform(case["lower"], case["upper"]).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

    def test_logpdf_edge_cases(self):
        test_cases = [
            {"lower": 1, "upper": 1, "x": 1, "expected": 0},
            {"lower": 0, "upper": 3, "x": 0, "expected": -np.log(4)},
            {"lower": 0, "upper": 3, "x": 3, "expected": -np.log(4)},
        ]
        for case in test_cases:
            logprob = DiscreteUniform(case["lower"], case["upper"]).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

    #
    # SAMPLING SHAPE
    #

    #
    # LOGPDF SHAPES
    #


def discreteuniform_mean(lower, upper):
    return (lower + upper) / 2.0


def discreteuniform_variance(lower, upper):
    return (np.pow((upper - lower + 1.0), 2) - 1.0) / 12.0


if __name__ == "__main__":
    unittest.main()
