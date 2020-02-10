"""Integrators for Hamiltonian and Riemanian trajectories.

.. note:
    This file is a "flat zone": position and momentum are one-dimensional arrays.
    Any raveling/unraveling logic must be placed at a higher level.
"""
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as np
import jax.numpy.DeviceArray as Array


__all__ = ["leapfrog_integrator"]


@partial(jax.jit, static_argnums=(0,))
def leapfrog_integrator(
    potential: Callable,
    path_length: float,
    step_size: Array,
    position: Array,
    momentum: Array,
    potential_grad: Array,
) -> Tuple[Array, Array, Array, Array]:
    """Second order symplectic integrator that uses the leapfrog algorithm.
    """
    position, momentum = np.copy(position), np.copy(momentum)
    momentum -= step_size * potential_grad / 2  # half step
    for _ in np.arange(np.round(path_length / step_size) - 1):
        position = position + step_size * momentum  # whole step
        potential_value, potential_grad = potential(position)
        momentum = momentum - step_size * potential_grad  # whole step
    position = position + step_size * momentum  # whole step
    potential_value, potential_grad = potential(position)
    momentum = momentum - step_size * potential_grad / 2  # half step

    return position, -momentum, potential_value, potential_grad
