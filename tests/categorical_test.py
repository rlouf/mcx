import unittest

import scipy
from jax import numpy as np
from jax import random

from mcx.distributions import Categorical


class CategoricalTest(unittest.TestCase):
    def setUp(self):
        self.rng_key = random.PRNGKey(0)

    #
    # SAMPLING CORRECTNESS
    #

    def test_sample_mode(self):
        test_cases = [
            {"probs": np.array([0, 1.0]), "mode": 1},
            {"probs": np.array([0.501, 0.499]), "mode": 0},
            {"probs": np.array([0.2, 0.7, 0.1]), "mode": 1},
            {"probs": np.array([1.0]), "mode": 0},
        ]
        for case in test_cases:
            samples = (
                Categorical(case["probs"]).sample(self.rng_key, (100_000,)).__array__()
            )
            mode = scipy.stats.mode(samples, axis=0)
            self.assertEqual(mode, case["mode"])

    #
    # LOGPDF CORRECTNESS
    #

    def test_logpdf_out_of_support(self):
        test_cases = [
            {"probs": np.array([0.1, 0.2, 0.7]), "x": -1, "expected": -np.inf},
            {"probs": np.array([0.1, 0.2, 0.7]), "x": 3, "expected": -np.inf},
            {"probs": np.array([0.1, 0.2, 0.7]), "x": 3.5, "expected": -np.inf},
        ]
        for case in test_cases:
            logprob = Categorical(case["probs"]).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

    def test_logpdf_edge_cases(self):
        test_cases = [
            {"probs": np.array([1]), "x": 0, "expected": 0.0},
            {"probs": np.array([0, 1]), "x": 1, "expected": 0},
            {"probs": np.array([0, 1]), "x": 0, "expected": -np.inf},
        ]
        for case in test_cases:
            logprob = Categorical(case["probs"]).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

    def test_logpdf_sanity_checks(self):
        test_cases = []
        for case in test_cases:
            logprob = Categorical(case["probs"]).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

    #
    # SAMPLING SHAPE
    #

    def test_sample_single_parameter(self):
        test_cases = [
            {"sample_shape": (), "expected_shape": (1,)},
            {"sample_shape": (100,), "expected_shape": (100, 1)},
            {"sample_shape": (100, 10), "expected_shape": (100, 10, 1)},
            {"sample_shape": (1, 100), "expected_shape": (1, 100, 1)},
        ]
        for case in test_cases:
            samples = Categorical(np.array([0.2, 0.7, 0.1])).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_array_of_parameters(self):
        test_cases = [
            {
                "probs": np.array([[0.1, 0.9], [0.2, 0.3], [0.5, 0.5]]),
                "sample_shape": (),
                "expected_shape": (3,),
            },
            {
                "probs": np.array([[0.1, 0.9], [0.2, 0.3], [0.5, 0.5]]),
                "sample_shape": (100,),
                "expected_shape": (100, 3),
            },
            {
                "probs": np.array([[0.1, 0.9], [0.2, 0.3], [0.5, 0.5]]),
                "sample_shape": (100, 10),
                "expected_shape": (100, 10, 3),
            },
            {
                "probs": np.array(
                    [
                        [[0.1, 0.9], [0.2, 0.3], [0.5, 0.5]],
                        [[0.1, 0.9], [0.2, 0.3], [0.5, 0.5]],
                    ]
                ),
                "sample_shape": (100, 10),
                "expected_shape": (100, 10, 2, 3),
            },
        ]
        for case in test_cases:
            samples = Categorical(case["probs"]).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    #
    # LOGPDF SHAPES
    #

    def test_logpdf_shape(self):
        test_cases = [
            {"x": 0, "probs": np.array([0.4, 0.1, 0.5]), "expected_shape": ()},
            {
                "x": 1,
                "probs": np.array([[0.1, 0.9], [0.2, 0.8]]),
                "expected_shape": (2,),
            },
        ]
        for case in test_cases:
            log_prob = Categorical(case["probs"]).logpdf(case["x"])
            self.assertEqual(log_prob.shape, case["expected_shape"])


if __name__ == "__main__":
    unittest.main()
