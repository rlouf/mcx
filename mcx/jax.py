"""JAX-related utilities."""
from collections import namedtuple

import jax.lax as lax
import jax.numpy as np
from jax import jit
from jax.dtypes import canonicalize_dtype
from jax.experimental import host_callback
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten

__all__ = ["ravel_pytree", "wait_until_computed"]


pytree_metadata = namedtuple("pytree_metadata", ["flat", "shape", "size", "dtype"])


def wait_until_computed(x):
    """Wait until all the elements of x have been computed.

    This is useful to display accurate computation times when using
    lax.scan, for instance.
    """
    for leaf in tree_leaves(x):
        leaf.block_until_ready()


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


def progress_bar_factory(tqdm_pbar, print_rate):
    """Factory that builds a progress bar decorator"""

    def _update_tqdm(arg, _):
        tqdm_pbar.update(arg)

    @jit
    def _update_progress_bar(arg, result):
        """Updates tqdm progress bar of a scan/loop only if the iteration
        number is a multiple of the print_rate
        """
        iter_num = arg

        result = lax.cond(
            iter_num % print_rate == 0,
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=result),
            lambda _: result,
            operand=None,
        )
        return result

    def progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must be looping over a tuple who's first element is `np.arange(num_samples)`.
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            "x is a tuple: (iteration_number, key)"
            iter_num = x[0]
            _ = _update_progress_bar(iter_num, iter_num)
            return func(carry, x)

        return wrapper_progress_bar

    return progress_bar_scan
