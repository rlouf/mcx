import unittest

from jax import numpy as np
from jax import random

from mcx.distributions import Uniform


class UniformTest(unittest.TestCase):
    def setUp(self):
        self.rng_key = random.PRNGKey(0)

    #
    # SAMPLING CORRECTNESS
    #

    #
    # LOGPDF CORRECTNESS
    #

    def test_logpdf_out_of_support(self):
        test_cases = [
            {"x": 0, "expected": 0.0},  # boundary belongs to support
            {"x": 1, "expected": 0.0},  # boundary belongs to support
            {"x": -0.01, "expected": -np.inf},
            {"x": 1.001, "expected": -np.inf},
        ]
        for case in test_cases:
            logprob = Uniform(0, 1).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

    #
    # SAMPLING SHAPE
    #

    #
    # LOGPDF SHAPES
    #


if __name__ == "__main__":
    unittest.main()
