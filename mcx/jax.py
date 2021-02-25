"""JAX-related utilities."""
from collections import namedtuple

import jax.lax as lax
import jax.numpy as jnp
from jax import jit
from jax.dtypes import canonicalize_dtype
from jax.experimental import host_callback
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten

__all__ = ["choice", "ravel_pytree", "wait_until_computed"]


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
            jnp.ravel(l), jnp.shape(l), jnp.size(l), canonicalize_dtype(lax.dtype(l))
        ),
        leaves,
    )
    leaves_idx = jnp.cumsum(jnp.array((0,) + tuple(d.size for d in leaves_metadata)))

    def unravel_list(arr):
        return [
            jnp.reshape(
                lax.dynamic_slice_in_dim(arr, leaves_idx[i], m.size), m.shape
            ).astype(m.dtype)
            for i, m in enumerate(leaves_metadata)
        ]

    flat = (
        jnp.concatenate([m.flat for m in leaves_metadata])
        if leaves_metadata
        else jnp.array([])
    )

    return flat, unravel_list


def progress_bar_factory(tqdm_pbar, num_samples):
    """Factory that builds a progress bar decorator"""

    if num_samples > 20:
        print_rate = int(num_samples / 20)
    else:
        print_rate = 1

    remainder = num_samples % print_rate

    def _update_tqdm(arg, _):
        tqdm_pbar.update(arg)

    @jit
    def _update_progress_bar(iter_num):
        """Updates tqdm progress bar of a scan/loop only if the iteration
        number is a multiple of the print_rate
        """
        _ = lax.cond(
            (iter_num % print_rate == 0) & (iter_num != num_samples - remainder),
            lambda _: host_callback.id_tap(_update_tqdm, print_rate, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

        _ = lax.cond(
            iter_num == num_samples - remainder,
            lambda _: host_callback.id_tap(_update_tqdm, remainder, result=iter_num),
            lambda _: iter_num,
            operand=None,
        )

    def progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must be looping over a tuple who's first element is `np.arange(num_samples)`.
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            "x is a tuple: (iteration_number, key)"
            iter_num = x[0]
            _update_progress_bar(iter_num)
            return func(carry, x)

        return wrapper_progress_bar

    return progress_bar_scan
