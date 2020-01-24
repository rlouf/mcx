import unittest

import jax
from jax import numpy as np
from numpy.testing import assert_array_almost_equal

from mcmx.distributions import Bernoulli


class BernoulliTest(unittest.TestCase):
    def setUp(self):
        self.rng_key = jax.random.PRNGKey(0)

    #
    # SAMPLING CORRECTNESS
    #

    def test_sample_frequency(self):
        probabilities = [0, 1, 0.5]
        for p in probabilities:
            samples = Bernoulli(p).sample(self.rng_key, (1_000_000,))
            avg = np.mean(samples, axis=0).item()
            self.assertAlmostEqual(avg, p, places=3)

    def test_sample_frequency_vectorized(self):
        probabilities = np.array([0, 1, 0.5])
        samples = Bernoulli(probabilities).sample(self.rng_key, (1_000_000,))
        avg = np.mean(samples, axis=0).__array__()
        assert_array_almost_equal(probabilities, avg, decimal=3)

    #
    # LOGPDF CORRECTNESS
    #

    def test_logpdf_edge_cases(self):
        test_cases = [
            {"p": 1, "x": 0, "expected": -np.inf},
            {"p": 0, "x": 1, "expected": -np.inf},
        ]
        for case in test_cases:
            logprob = Bernoulli(case["p"]).logpdf(case["x"])
            self.assertEqual(logprob.item(), case["expected"])

    def test_logpdf_out_of_support(self):
        test_cases = [
            {"p": 0.5, "x": 0.5, "expected": -np.inf},
            {"p": 0.5, "x": 3, "expected": -np.inf},
            {"p": 0.5, "x": -1.1, "expected": -np.inf},
        ]
        for case in test_cases:
            logprob = Bernoulli(case["p"]).logpdf(case["x"])
            self.assertEqual(logprob.item(), case["expected"])

    def test_logpdf_few_cases(self):
        """Values obtained from `scipy.stats.bernoulli.logpmf`

        Note:
            Comparison fails at the 8th decimal place.
        """
        test_cases = [
            {"p": 0.5, "x": 0, "expected": -0.6931471805599453},
            {"p": 0.5, "x": 1, "expected": -0.6931471805599453},
            {"p": 0.2, "x": 1, "expected": -1.6094379124341003},
            {"p": 0.2, "x": 0, "expected": -0.22314355131420976},
            {"p": 0.7, "x": 1, "expected": -0.35667494393873245},
            {"p": 0.7, "x": 0, "expected": -1.203972804325936},
        ]
        for case in test_cases:
            logprob = Bernoulli(case["p"]).logpdf(case["x"])
            self.assertAlmostEqual(logprob.item(), case["expected"], places=7)

    #
    # SAMPLING SHAPES
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
            samples = Bernoulli(0.5).sample(self.rng_key, case["sample_shape"])
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_shape_array_parameter(self):
        test_cases = [
            {"p": np.array([0.3, 0.5]), "sample_shape": (), "expected_shape": (2,)},
            {
                "p": np.array([[0.3, 0.5], [0.1, 0.2]]),
                "sample_shape": (),
                "expected_shape": (2, 2),
            },
            {
                "p": np.array([0.3, 0.5]),
                "sample_shape": (10,),
                "expected_shape": (10, 2),
            },
            {
                "p": np.array([[0.3, 0.5], [0.1, 0.2]]),
                "sample_shape": (10,),
                "expected_shape": (10, 2, 2),
            },
            {
                "p": np.array([0.3, 0.5]),
                "sample_shape": (10, 10),
                "expected_shape": (10, 10, 2),
            },
            {
                "p": np.array([[0.3, 0.5], [0.1, 0.2]]),
                "sample_shape": (10, 10),
                "expected_shape": (10, 10, 2, 2),
            },
        ]
        for case in test_cases:
            samples = Bernoulli(case["p"]).sample(self.rng_key, case["sample_shape"])
            self.assertEqual(samples.shape, case["expected_shape"])

    #
    # LOGPDF SHAPES
    #

    def test_logpdf_shape(self):
        test_cases = [
            {"x": 1, "p": 0, "expected_shape": ()},
            {"x": 1, "p": np.array([0.1, 0.2]), "expected_shape": (2,)},
        ]
        for case in test_cases:
            log_prob = Bernoulli(case["p"]).logpdf(case["x"])
            self.assertEqual(log_prob.shape, case["expected_shape"])


if __name__ == "__main__":
    unittest.main()
