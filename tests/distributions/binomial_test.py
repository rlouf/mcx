import unittest

from jax import numpy as np
from jax import random
from scipy.stats import binom as scipy_binom

from mcx.distributions import Binomial


class BinomialTest(unittest.TestCase):
    def setUp(self):
        self.rng_key = random.PRNGKey(0)

    #
    # SAMPLING CORRECTNESS
    #

    def test_sample_mean(self):
        test_cases = [
            {"p": 0.1, "n": 100, "expected": binomial_mean(0.1, 100)},
            {"p": 0.5, "n": 1, "expected": binomial_mean(0.5, 1)},
            {"p": 0.9, "n": 10, "expected": binomial_mean(0.9, 10)},
            {"p": 0.9, "n": 100, "expected": binomial_mean(0.9, 100)},
        ]
        for case in test_cases:
            samples = Binomial(case["p"], case["n"]).sample(self.rng_key, (1_000_000,))
            avg = np.mean(samples, axis=0).item()
            self.assertAlmostEqual(avg, case["expected"], places=1)

    def test_sample_variance(self):
        test_cases = [
            {"p": 0.1, "n": 10, "expected": binomial_variance(0.1, 10)},
            {"p": 0.5, "n": 1, "expected": binomial_variance(0.5, 1)},
            {"p": 0.9, "n": 10, "expected": binomial_variance(0.9, 10)},
            {"p": 0.9, "n": 20, "expected": binomial_variance(0.9, 20)},
        ]
        for case in test_cases:
            samples = Binomial(case["p"], case["n"]).sample(self.rng_key, (1_000_000,))
            var = np.var(samples, axis=0).item()
            self.assertAlmostEqual(var, case["expected"], places=2)

    #
    # LOGPDF CORRECTNESS
    #

    def test_logpdf_out_of_support(self):
        test_cases = [
            {"p": 0.5, "n": 10, "x": 11, "expected": -np.inf},  # > 0
            {"p": 0.5, "n": 10, "x": -1, "expected": -np.inf},  # < 0
            {"p": 0.5, "n": 10, "x": 1.1, "expected": -np.inf},  # is integer
        ]
        for case in test_cases:
            logprob = Binomial(case["p"], case["n"]).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

    def test_logpdf_edge_cases(self):
        """Test following edge cases:

        - The probability of obtaining a non-zero number of successes with a
          probability of success of 0 should be 0;
        - The probability of not obtaining `n` with a probability of success of
          1 should be 0;
        - Conversely, the probability of obtaining 0 successes with a probability
          of success equal to 0 should be 1.
        - And the probability of obtaining `n` successes with a probability of
          success equal to 1 is 1.
        """
        test_cases = [
            {"p": 0, "n": 10, "x": 1, "expected": -np.inf},
            {"p": 1, "n": 10, "x": 1, "expected": -np.inf},
            {"p": 1, "n": 10, "x": 0, "expected": -np.inf},
            {"p": 1.0, "n": 10, "x": 10, "expected": 0.0},
            {"p": 0.0, "n": 10, "x": 0, "expected": 0.0},
        ]
        for case in test_cases:
            logprob = Binomial(case["p"], case["n"]).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

    def test_logpdf_compare_scipy(self):
        test_cases = [
            {"p": 0.5, "n": 10, "x": 5, "expected": scipy_binom.logpmf(5, 10, 0.5)},
            {
                "p": 0.9999,
                "n": 10,
                "x": 5,
                "expected": scipy_binom.logpmf(5, 10, 0.9999),
            },
            {
                "p": 0.0001,
                "n": 10,
                "x": 5,
                "expected": scipy_binom.logpmf(5, 10, 0.0001),
            },
        ]
        for case in test_cases:
            logprob = Binomial(case["p"], case["n"]).logpdf(case["x"])
            self.assertAlmostEqual(logprob, case["expected"], places=2)

    #
    # SAMPLING SHAPE
    #

    def test_sample_shape_scalar_arguments(self):
        """Test the correctness of broadcasting when both arguments are
        scalars. They are tested separately as scalars are an edge case when
        it comes to broadcasting.

        The trailing `1` stands for the batch size.
        """
        test_cases = [
            {"sample_shape": (), "expected_shape": (1,)},
            {"sample_shape": (100,), "expected_shape": (100, 1)},
            {"sample_shape": (100, 10), "expected_shape": (100, 10, 1)},
            {"sample_shape": (1, 100), "expected_shape": (1, 100, 1)},
        ]
        for case in test_cases:
            samples = Binomial(0.5, 10).sample(self.rng_key, case["sample_shape"])
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_shape_array_arguments_no_sample_shape(self):
        """Test the correctness of broadcasting when arguments can be arrays."""
        test_cases = [
            {
                "p": 0.5,
                "n": np.array([1, 2, 3, 4]),
                "sample_shape": (),
                "expected_shape": (4,),
            },
            {
                "p": np.array([0.1, 0.2, 0.3]),
                "n": 5,
                "sample_shape": (),
                "expected_shape": (3,),
            },
            {
                "p": 0.5,
                "n": np.array([[1, 2], [3, 4]]),
                "sample_shape": (),
                "expected_shape": (2, 2),
            },
            {
                "p": np.array([0.1, 0.2]),
                "n": np.array([[1, 2], [3, 4]]),
                "sample_shape": (),
                "expected_shape": (2, 2),
            },
        ]
        for case in test_cases:
            samples = Binomial(case["p"], case["n"]).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_shape_array_arguments_1d_sample_shape(self):
        test_cases = [
            {
                "p": 0.1,
                "n": np.array([1, 2, 3]),
                "sample_shape": (100,),
                "expected_shape": (100, 3),
            },
            {
                "p": np.array([0.1, 0.2, 0.3]),
                "n": 1,
                "sample_shape": (100,),
                "expected_shape": (100, 3),
            },
            {
                "p": 0.1,
                "n": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100,),
                "expected_shape": (100, 2, 2),
            },
            {
                "p": np.array([0.1, 0.2]),
                "n": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100,),
                "expected_shape": (100, 2, 2),
            },
        ]
        for case in test_cases:
            samples = Binomial(case["p"], case["n"]).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_shape_array_arguments_2d_sample_shape(self):
        test_cases = [
            {
                "p": 0.1,
                "n": np.array([1, 2, 3]),
                "sample_shape": (100, 2),
                "expected_shape": (100, 2, 3),
            },
            {
                "p": np.array([0.1, 0.2, 0.3]),
                "n": 1,
                "sample_shape": (100, 3),
                "expected_shape": (100, 3, 3),
            },
            {
                "p": 0.1,
                "n": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100, 2),
                "expected_shape": (100, 2, 2, 2),
            },
            {
                "p": np.array([0.1, 0.2]),
                "n": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100, 2),
                "expected_shape": (100, 2, 2, 2),
            },
        ]
        for case in test_cases:
            samples = Binomial(case["p"], case["n"]).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    #
    # LOGPDF SHAPES
    #

    def test_logpdf_shape(self):
        test_cases = [
            {"x": 2, "p": 0.5, "n": 10, "expected_shape": ()},
            {"x": 2, "p": np.array([0.1, 0.2]), "n": 10, "expected_shape": (2,)},
        ]
        for case in test_cases:
            log_prob = Binomial(case["p"], case["n"]).logpdf(case["x"])
            self.assertEqual(log_prob.shape, case["expected_shape"])


def binomial_mean(p, n):
    return p * n


def binomial_variance(p, n):
    return n * p * (1.0 - p)


if __name__ == "__main__":
    unittest.main()
