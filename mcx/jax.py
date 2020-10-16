"""JAX-related utilities."""
from collections import namedtuple

import jax.lax as lax
import jax.numpy as np
from jax.dtypes import canonicalize_dtype
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

__all__ = ["ravel_pytree"]


pytree_metadata = namedtuple("pytree_metadata", ["flat", "shape", "size", "dtype"])


# Sourced from numpyro.distributions.utils.py
# Copyright Contributors to the NumPyro project.
# SPDX-License-Identifier: Apache-2.0
def ravel_pytree(pytree):
    """Ravel pytrees.

    JAX's version of `ravel_pytree` uses `vjp` and therefore does not support
    some python types such as booleans. This function and `unravel_list` are
    found in the numpyro repository [1]_.

    .. [1]: Numpyro: https://github.com/pyro-ppl/numpyro

    """
    leaves, treedef = tree_flatten(pytree)
    flat, unravel_list = _ravel_list(*leaves)

    def unravel_pytree(arr):
        return tree_unflatten(treedef, unravel_list(arr))

    return flat, unravel_pytree


# Sourced from numpyro.distributions.utils.py
# Copyright Contributors to the NumPyro project.
# SPDX-License-Identifier: Apache-2.0
def _ravel_list(*leaves):
    leaves_metadata = tree_map(
        lambda l: pytree_metadata(
            np.ravel(l), np.shape(l), np.size(l), canonicalize_dtype(lax.dtype(l))
        ),
        leaves,
    )
    leaves_idx = np.cumsum(np.array((0,) + tuple(d.size for d in leaves_metadata)))

    def unravel_list(arr):
        return [
            np.reshape(
                lax.dynamic_slice_in_dim(arr, leaves_idx[i], m.size), m.shape
            ).astype(m.dtype)
            for i, m in enumerate(leaves_metadata)
        ]

    flat = (
        np.concatenate([m.flat for m in leaves_metadata])
        if leaves_metadata
        else np.array([])
    )

    return flat, unravel_list
