import unittest

from jax import numpy as np
from jax import random

from mcx.distributions import LogNormal


class LogNormalTest(unittest.TestCase):
    def setUp(self):
        self.rng_key = random.PRNGKey(0)

    #
    # SAMPLING CORRECTNESS
    #

    #
    # LOGPDF CORRECTNESS
    #

    def test_out_of_support(self):
        test_cases = [{"x": 0, "expected": -np.inf}, {"x": -1, "expected": -np.inf}]
        for case in test_cases:
            logprob = LogNormal(0, 1).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

    #
    # SAMPLING SHAPE
    #

    #
    # LOGPDF SHAPES
    #


if __name__ == "__main__":
    unittest.main()
