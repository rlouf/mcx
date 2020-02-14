"""Integrators for Hamiltonian and Riemanian trajectories.

.. note:
    This file is a "flat zone": position and momentum are one-dimensional arrays.
    Any raveling/unraveling logic must be placed at a higher level.
"""
from typing import Callable, Tuple

import jax
import jax.numpy as np
import jax.numpy.DeviceArray as Array


__all__ = ["leapfrog_integrator"]

IntegratorState = Tuple[Array, Array, float, float]


def leapfrog_integrator(
    logpdf: Callable, path_length: float, step_size: float,
) -> Callable:
    @jax.jit
    def step(_, state: IntegratorState) -> IntegratorState:
        """Second order symplectic integrator that uses the leapfrog algorithm.
        """
        position, momentum, _, log_prob_grad = state

        momentum -= step_size * log_prob_grad / 2  # half step
        for _ in np.arange(np.round(path_length / step_size) - 1):
            position = position + step_size * momentum  # whole step
            log_prob, log_prob_grad = logpdf(position)
            momentum = momentum - step_size * log_prob_grad  # whole step
        position = position + step_size * momentum  # whole step
        log_prob, log_prob_grad = logpdf(position)
        momentum = momentum - step_size * log_prob_grad / 2  # half step

        return (position, momentum, log_prob, log_prob_grad)

    return step
