"""Integrators for Hamiltonian and Riemanian trajectories.
"""
from functools import partial

import jax
from jax import numpy as np


@partial(jax.jit, static_argnums=(4,))
def leapfrog_integrator(
    position, momentum, potential, potential_grad, path_length, step_size
):
    """Second order symplectic integrator that uses the leapfrog algorithm
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
