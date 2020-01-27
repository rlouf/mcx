import unittest

from jax import numpy as np
from jax import random

from mcx.distributions import Beta


class BetaTest(unittest.TestCase):
    def setUp(self):
        self.rng_key = random.PRNGKey(0)

    #
    # SAMPLING CORRECTNESS
    #

    def test_sample_mean(self):
        test_cases = [
            {"a": 1, "b": 1, "expected": beta_mean(1, 1)},
            {"a": 0.1, "b": 1, "expected": beta_mean(0.1, 1)},
            {"a": 10, "b": 10, "expected": beta_mean(10, 10)},
            {"a": 10, "b": 100, "expected": beta_mean(10, 100)},
        ]
        for case in test_cases:
            samples = Beta(case["a"], case["b"]).sample(self.rng_key, (100_000,))
            avg = np.mean(samples, axis=0).item()
            self.assertAlmostEqual(avg, case["expected"], places=2)

    def test_sample_variance(self):
        test_cases = [
            {"a": 1, "b": 1, "expected": beta_variance(1, 1)},
            {"a": 0.1, "b": 1, "expected": beta_variance(0.1, 1)},
            {"a": 10, "b": 10, "expected": beta_variance(10, 10)},
            {"a": 10, "b": 100, "expected": beta_variance(10, 100)},
        ]
        for case in test_cases:
            samples = Beta(case["a"], case["b"]).sample(self.rng_key, (100_000,))
            var = np.var(samples, axis=0).item()
            self.assertAlmostEqual(var, case["expected"], places=2)

    #
    # LOGPDF CORRECTNESS
    # We decice here to trust the implementation in `jax.scipy.stats`, but
    # still need to test that the logpdf is limited by the support.
    #

    def test_logpdf_out_of_support(self):
        test_cases = [
            {"x": -0.1, "expected": -np.inf},
            {"x": 1.1, "expected": -np.inf},
            {
                "x": 0,
                "expected": -np.inf,
            },  # the logpdf is defined on the *open* interval ]0, 1[
            {"x": 1, "expected": -np.inf},  # idem
        ]
        for case in test_cases:
            logprob = Beta(1, 1).logpdf(case["x"])
            self.assertEqual(logprob, case["expected"])

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
            samples = Beta(1, 1).sample(self.rng_key, case["sample_shape"])
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_shape_array_arguments_no_sample_shape(self):
        """Test the correctness of broadcasting when arguments can be arrays."""
        test_cases = [
            {
                "a": 1,
                "b": np.array([1, 2, 3]),
                "sample_shape": (),
                "expected_shape": (3,),
            },
            {
                "a": np.array([1, 2, 3]),
                "b": 1,
                "sample_shape": (),
                "expected_shape": (3,),
            },
            {
                "a": 1,
                "b": np.array([[1, 2], [3, 4]]),
                "sample_shape": (),
                "expected_shape": (2, 2),
            },
            {
                "a": np.array([1, 2]),
                "b": np.array([[1, 2], [3, 4]]),
                "sample_shape": (),
                "expected_shape": (2, 2),
            },
        ]
        for case in test_cases:
            samples = Beta(case["a"], case["b"]).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_shape_array_arguments_1d_sample_shape(self):
        test_cases = [
            {
                "a": 1,
                "b": np.array([1, 2, 3]),
                "sample_shape": (100,),
                "expected_shape": (100, 3),
            },
            {
                "a": np.array([1, 2, 3]),
                "b": 1,
                "sample_shape": (100,),
                "expected_shape": (100, 3),
            },
            {
                "a": 1,
                "b": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100,),
                "expected_shape": (100, 2, 2),
            },
            {
                "a": np.array([1, 2]),
                "b": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100,),
                "expected_shape": (100, 2, 2),
            },
        ]
        for case in test_cases:
            samples = Beta(case["a"], case["b"]).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    def test_sample_shape_array_arguments_2d_sample_shape(self):
        test_cases = [
            {
                "a": 1,
                "b": np.array([1, 2, 3]),
                "sample_shape": (100, 2),
                "expected_shape": (100, 2, 3),
            },
            {
                "a": np.array([1, 2, 3]),
                "b": 1,
                "sample_shape": (100, 3),
                "expected_shape": (100, 3, 3),
            },
            {
                "a": 1,
                "b": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100, 2),
                "expected_shape": (100, 2, 2, 2),
            },
            {
                "a": np.array([1, 2]),
                "b": np.array([[1, 2], [3, 4]]),
                "sample_shape": (100, 2),
                "expected_shape": (100, 2, 2, 2),
            },
        ]
        for case in test_cases:
            samples = Beta(case["a"], case["b"]).sample(
                self.rng_key, case["sample_shape"]
            )
            self.assertEqual(samples.shape, case["expected_shape"])

    #
    # LOGPDF SHAPES
    #

    def test_logpdf_shape(self):
        test_cases = [
            {"x": 0.5, "a": 0, "b": 1, "expected_shape": ()},
            {"x": 0.5, "a": np.array([1, 2]), "b": 1, "expected_shape": (2,)},
        ]
        for case in test_cases:
            log_prob = Beta(case["a"], case["b"]).logpdf(case["x"])
            self.assertEqual(log_prob.shape, case["expected_shape"])


def beta_mean(a, b):
    return a / (a + b)


def beta_variance(a, b):
    return (a * b) / ((a + b) ** 2 * (a + b + 1))


if __name__ == "__main__":
    unittest.main()
