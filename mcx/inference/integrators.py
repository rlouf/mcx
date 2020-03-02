"""Integrators of hamiltonian trajectories on euclidean and riemannian
manifolds.

.. note:
    This file is a "flat zone": position and momentum are one-dimensional arrays.
    Any raveling/unraveling logic must be placed at a higher level.
"""
from functools import partial
from typing import Callable, NamedTuple

import jax
from jax import numpy as np
from jax.numpy import DeviceArray as Array


__all__ = ["hmc_integrator", "empirical_hmc_integrator", "velocity_verlet"]


class IntegratorState(NamedTuple):
    position: Array
    momentum: Array
    log_prob: float
    log_prob_grad: float


Integrator = Callable[[jax.random.PRNGKey, IntegratorState], IntegratorState]
IntegratorStep = Callable[[IntegratorState, float], IntegratorState]


def hmc_integrator(
    integrator_step: IntegratorStep, step_size: float, path_length: float = 1.0
) -> Integrator:
    """Vanilla HMC integrator.

    Given a path length and a step size, the HMC integrator will run the appropriate
    number of iteration steps (typically with the velocity Verlet algorithm).

    When `path_length` = `step_size` the integrator reduces to the integrator
    step.
    """

    num_steps = np.clip(path_length / step_size, a_min=1).astype(int)

    @jax.jit
    def integrate(_, state: IntegratorState) -> IntegratorState:
        new_state = jax.lax.fori_loop(
            0, num_steps, lambda i, state: integrator_step(state, step_size), state
        )
        return new_state

    return integrate


def empirical_hmc_integrator(
    integrator_step: IntegratorStep, path_length_generator: Callable, step_size: float
) -> Integrator:
    """Integrator for the empirical HMC algorithm.

    The empirical HMC algorithm [1]_ uses an adaptive scheme for the path
    length: during warmup, a distribution of eligible path lengths is computed;
    the integrator draws a random path length value from this distribution each
    time it is called.

    References
    ----------
    .. [1]: Wu, Changye, Julien Stoehr, and Christian P. Robert. "Faster
            Hamiltonian Monte Carlo by learning leapfrog scale." arXiv preprint
            arXiv:1810.04449 (2018).
    """

    @jax.jit
    def integrate(
        rng_key: jax.random.PRNGKey, state: IntegratorState
    ) -> IntegratorState:
        path_length = path_length_generator(rng_key)
        num_steps = np.clip(path_length / step_size, a_min=1).astype(int)
        new_state = jax.lax.fori_loop(
            0, num_steps, lambda i, state: integrator_step(state, step_size), state
        )
        return new_state

    return integrate


def velocity_verlet(logpdf: Callable, kinetic_energy_fn: Callable) -> IntegratorStep:
    """The velocity Verlet integrator [1]_.

    References
    ----------
    .. [1]: Bou-Rabee, Nawaf, and Jesús Marıa Sanz-Serna. "Geometric
            integrators and the Hamiltonian Monte Carlo method." Acta Numerica 27
            (2018): 113-206.
    """

    @partial(jax.jit, static_argnums=(1,))
    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, _, log_prob_grad = state

        momentum = momentum - 0.5 * step_size * log_prob_grad  # half step
        kinetic_grad = jax.grad(kinetic_energy_fn)(momentum)
        position = position + step_size * kinetic_grad  # whole step
        log_prob, log_prob_grad = jax.value_and_grad(logpdf)(position)
        momentum = momentum - 0.5 * step_size * log_prob_grad  # half step

        return IntegratorState(position, momentum, log_prob, log_prob_grad)

    return one_step
