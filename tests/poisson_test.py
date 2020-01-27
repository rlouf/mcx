import unittest

from jax import numpy as np
from jax import random

from mcmx.distributions import Poisson


class PoissonTest(unittest.TestCase):
    def setUp(self):
        self.rng_key = random.PRNGKey(0)

    #
    # SAMPLING CORRECTNESS
    #

    #
    # LOGPDF CORRECTNESS
    #

    def test_logpdf_edge_cases(self):
        # Should work with ints
        pass

    def test_logpdf_out_of_support(self):
        test_cases = [{"x": -1.0, "expected": -np.inf}, {"x": 1.1, "expected": -np.inf}]
        for case in test_cases:
            logprob = Poisson(1.0).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

    #
    # SAMPLING SHAPE
    #

    #
    # LOGPDF SHAPES
    #


if __name__ == "__main__":
    unittest.main()
