"""Adaptive algorithms for Markov Chain Monte Carlo.

This is a collection of re-usable adaptive schemes for monte carlo algorithms.
The algorithms are used during the warm-up phase of the inference and are
decorrelated from any particular algorithm (dynamic HMC's adaptive choice of
path length is not included, for instance).

The Stan Manual [1]_ is a very good reference on automatic tuning of
parameters used in Hamiltonian Monte Carlo.

.. [1]: "HMC Algorithm Parameters", Stan Manual
        https://mc-stan.org/docs/2_20/reference-manual/hmc-algorithm-parameters.html
"""
from functools import partial
from typing import Callable, Tuple

import jax
from jax import numpy as np

__all__ = ["longest_batch_before_turn"]


def longest_batch_before_turn(integrator_step: Callable) -> Callable:
    """Learn the number of steps one can make before the trajectory makes a
    U-Turn. This routine is part of the adaptive strategy described in [1]_:
    during the warmup phase we run this scheme many times in order to get a
    distribution of numbers of steps before U-Turn. We then sample elements
    from this distribution during inference to use as the number of integration
    steps.

    References
    ----------
    .. [1]: Wu, Changye, Julien Stoehr, and Christian P. Robert. "Faster
            Hamiltonian Monte Carlo by learning leapfrog scale." arXiv preprint
            arXiv:1810.04449 (2018).
    """

    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def run(
        initial_position: np.DeviceArray,
        initial_momentum: np.DeviceArray,
        step_size: float,
        num_integration_steps: int,
    ):
        def cond(state: Tuple) -> bool:
            iteration, position, momentum = state
            return is_u_turn or iteration == num_integration_steps

        def update(state: Tuple) -> Tuple:
            iteration, position, momentum = state
            iteration += 1
            position, momentum = integrator_step(position, momentum, step_size, 1)
            return (iteration, position, momentum)

        result = jax.lax.while_loop(
            cond, update, (0, initial_position, initial_momentum)
        )

        return result[0]

    return run


@partial(jax.jit, static_argnums=(0, 2))
def is_u_turn(
    initial_position: np.DeviceArray,
    position: np.DeviceArray,
    inverse_mass_matrix: np.DeviceArray,
    momentum: np.DeviceArray,
) -> bool:
    """Detect when the trajectory starts turning back towards the point
    where it started.
    """
    v = np.multiply(inverse_mass_matrix, momentum)
    position_vec = position - initial_position
    projection = np.multiply(position_vec, v)
    return np.where(projection < 0, True, False)
