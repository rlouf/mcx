import unittest

import jax
from jax import numpy as np
from numpy.testing import assert_array_almost_equal

from mcmx.distributions import Normal


class NormalTest(unittest.TestCase):
    def setUp(self):
        self.rng_key = jax.random.PRNGKey(0)

    #
    # SAMPLING CORRECTNESS
    #

    def test_sample_mean(self):
        means = [0, -1, 10, 100]
        for mu in means:
            samples = Normal(mu, 0.01).sample(self.rng_key, (1_000_000,))
            avg = np.mean(samples, axis=0).block_until_ready().item()
            self.assertAlmostEqual(avg, mu, places=4)

    def test_sample_mean_vectorized(self):
        means = np.array([0, -1, 10, 100])
        samples = Normal(means, 0.01).sample(self.rng_key, (100_000,))
        avg = np.mean(samples, axis=0).block_until_ready().__array__()
        assert_array_almost_equal(means, avg, decimal=4)

    def test_sample_std(self):
        std = [0.1, 1, 10]
        for sigma in std:
            samples = Normal(0, sigma).sample(self.rng_key, (1_000_000,))
            deviation = np.std(samples, axis=0).block_until_ready().item()
            self.assertAlmostEqual(deviation, sigma, places=1)

    #
    # LOGPDF CORRECTNESS
    # We trust JAX's implementation in scipy.stats.norm.logpdf
    #

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
            samples = Normal(0, 1).sample(self.rng_key, case["sample_shape"])
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_shape_array_arguments_no_sample_shape(self):
        """Test the correctness of broadcasting when arguments can be arrays."""
        test_cases = [
            {
                "mu": 1,
                "sigma": np.array([1, 2, 3]),
                "sample_shape": (),
                "expected_shape": (3,),
            },
            {
                "mu": np.array([1, 2, 3]),
                "sigma": 1,
                "sample_shape": (),
                "expected_shape": (3,),
            },
            {
                "mu": 1,
                "sigma": np.array([[1, 2], [3, 4]]),
                "sample_shape": (),
                "expected_shape": (2, 2),
            },
            {
                "mu": np.array([1, 2]),
                "sigma": np.array([[1, 2], [3, 4]]),
                "sample_shape": (),
                "expected_shape": (2, 2),
            },
        ]
        for case in test_cases:
            samples = Normal(case["mu"], case["sigma"]).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_shape_array_arguments_1d_sample_shape(self):
        test_cases = [
            {
                "mu": 1,
                "sigma": np.array([1, 2, 3]),
                "sample_shape": (100,),
                "expected_shape": (100, 3),
            },
            {
                "mu": np.array([1, 2, 3]),
                "sigma": 1,
                "sample_shape": (100,),
                "expected_shape": (100, 3),
            },
            {
                "mu": 1,
                "sigma": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100,),
                "expected_shape": (100, 2, 2),
            },
            {
                "mu": np.array([1, 2]),
                "sigma": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100,),
                "expected_shape": (100, 2, 2),
            },
        ]
        for case in test_cases:
            samples = Normal(case["mu"], case["sigma"]).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_shape_array_arguments_2d_sample_shape(self):
        test_cases = [
            {
                "mu": 1,
                "sigma": np.array([1, 2, 3]),
                "sample_shape": (100, 2),
                "expected_shape": (100, 2, 3),
            },
            {
                "mu": np.array([1, 2, 3]),
                "sigma": 1,
                "sample_shape": (100, 3),
                "expected_shape": (100, 3, 3),
            },
            {
                "mu": 1,
                "sigma": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100, 2),
                "expected_shape": (100, 2, 2, 2),
            },
            {
                "mu": np.array([1, 2]),
                "sigma": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100, 2),
                "expected_shape": (100, 2, 2, 2),
            },
        ]
        for case in test_cases:
            samples = Normal(case["mu"], case["sigma"]).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    #
    # LOGPDF SHAPES
    #

    def test_logpdf_shape(self):
        test_cases = [
            {"x": 1, "mu": 0, "sigma": 1, "expected_shape": ()},
            {"x": 1, "mu": np.array([1, 2]), "sigma": 1, "expected_shape": (2,)},
        ]
        for case in test_cases:
            log_prob = Normal(case["mu"], case["sigma"]).logpdf(case["x"])
            self.assertEqual(log_prob.shape, case["expected_shape"])


if __name__ == "__main__":
    unittest.main()
