"""Integrators for Hamiltonian and Riemanian trajectories.

.. note:
    This file is a "flat zone": position and momentum are one-dimensional arrays.
    Any raveling/unraveling logic must be placed at a higher level.
"""
from typing import Callable, NamedTuple

import jax
import jax.numpy as np
import jax.numpy.DeviceArray as Array


__all__ = ["hmc_integrator"]


class IntegratorState(NamedTuple):
    position: Array
    momentum: Array
    log_prob: float
    log_prob_grad: float


def hmc_integrator(
    integrator_step: Callable, path_length: float, step_size: float
) -> Callable:
    @jax.jit
    def integrate(_, state: IntegratorState) -> IntegratorState:
        state = integrator_step(state, step_size, path_length)
        return state

    return integrate


def ehmc_integrator(
    integrator_step: Callable, path_length_generator: Callable, step_size: float
) -> Callable:
    @jax.jit
    def integrate(rng_key, state: IntegratorState) -> IntegratorState:
        path_length = path_length_generator(rng_key)
        state = integrator_step(state, step_size, path_length)
        return state

    return integrate


def leapfrog(logpdf: Callable) -> Callable:
    @jax.jit
    def step(
        state: IntegratorState, step_size: float, path_length: float
    ) -> IntegratorState:
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

        return IntegratorState(position, momentum, log_prob, log_prob_grad)

    return step
