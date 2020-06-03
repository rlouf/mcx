from typing import Callable, Optional

import jax.numpy as np

import mcx.distributions as md
import trax.layers as tl


class Dense(tl.Dense, md.Distribution):
    """Dense layer which basically computes:

    ..math: `X * w + b`
    """

    def __init__(
        self,
        n_units: int,
        distribution: Optional[md.Distribution] = None,
        transform: Optional[Callable] = None,
        kernel_initializer=tl.init.GlorotUniformInitializer(),
        bias_initializer=tl.init.RandomNormalInitializer(1e-6),
    ):
        super(tl.Dense, self).__init__(n_units, kernel_initializer, bias_initializer)
        self.distribution = distribution
        self.transform = transform if transform else lambda x: x

    def forward(self, x, weights):
        w, b = weights
        w_tilde = self.transform(w)
        return np.dot(x, w_tilde) + b

    def sample(self, x, weights, rng_key, sample_shape):
        """Used during forward sample"""
        w, b = weights
        w_tilde = self.distribution.sample(
            rng_key, sample_shape
        )  # this may be problematic
        return np.dot(x, w_tilde) + b

    def logpdf(self, weights):
        w, b = weights
        return self.distribution.logpdf(w)
