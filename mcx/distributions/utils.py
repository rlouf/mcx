from jax import lax
from jax import numpy as np


def broadcast_batch_shape(*shapes):
    """Compute the batch shape by broadcasting the arguments.

    We use `lax.broadcast_shapes` to get the shape of broadcasted arguments. We
    default the batch shape to (1,) when the distribution is initiated with
    scalar values.

    To see why we need to do that, consider the following model:

        >>> def toy_model():
        ...     sigma = np.array([1, 2, 3])
        ...     x @ Normal(0, sigma) # shape (n_samples, 3)
        ...     q @ Normal(1, 1) # shape (n_samples,)
        ...     y @ Normal(x, q)

    When sampling, the last line will trigger a broadcasting error since
    Numpy's default is to broadcast (n,) to (1,n). To avoid this we explicit
    the fact that a distribution initiated with scalar arguments has a batch
    size of 1.
    """
    broadcasted_shape = lax.broadcast_shapes(*shapes)
    if len(broadcasted_shape) == 0:
        return (1,)
    return broadcasted_shape


# Sourced from numpyro.distributions.utils.py
# Copyright Contributors to the NumPyro project.
# SPDX-License-Identifier: Apache-2.0
def limit_to_support(logpdf):
    """Decorator that enforces the distrbution's support by returning `-np.inf`
    if the value passed to the logpdf is out of support.
    """

    def wrapper(self, *args):
        log_prob = logpdf(self, *args)
        value = args[0]
        mask = self.support(value)
        log_prob = np.where(mask, log_prob, -np.inf)
        return log_prob

    return wrapper
