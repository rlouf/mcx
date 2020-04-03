"""Integrators of hamiltonian trajectories on euclidean and riemannian
manifolds.

IntegratorStep does not need to compute and return the logprob!
"""
from functools import partial
from typing import Callable, NamedTuple

import jax
from jax import numpy as np
from jax.tree_util import tree_multimap
from jax.numpy import DeviceArray as Array


__all__ = [
    "empirical_hmc_proposal",
    "hmc_proposal",
    "four_stages_integrator",
    "mclachlan_integrator",
    "velocity_verlet",
    "yoshida_integrator",
]


class IntegratorState(NamedTuple):
    position: Array
    momentum: Array
    log_prob_grad: float


Proposal = IntegratorState


Proposer = Callable[[jax.random.PRNGKey, Proposal], Proposal]
Integrator = Callable[[IntegratorState, float], IntegratorState]


#
# Proposals
#


def hmc_proposal(
    integrator: Integrator, step_size: float, num_integration_steps: int = 1
) -> Proposer:
    """Vanilla HMC proposal.

    Given a path length and a step size, the HMC proposer will run the appropriate
    number of integration steps (typically with the velocity Verlet algorithm).

    When `num_integration_steps` = 1 the proposer reduces to the integrator
    step.
    """

    @jax.jit
    def propose(_, init_state: Proposal) -> Proposal:
        new_state = jax.lax.fori_loop(
            0, num_integration_steps, lambda i, state: integrator(state, step_size), init_state
        )
        return new_state

    return propose


def empirical_hmc_proposal(
    integrator: Integrator, path_length_generator: Callable, step_size: float
) -> Proposer:
    """Proposal for the empirical HMC algorithm.

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
    def propose(rng_key: jax.random.PRNGKey, state: Proposal) -> Proposal:
        path_length = path_length_generator(rng_key)
        num_steps = np.clip(path_length / step_size, a_min=1).astype(int)
        new_state = jax.lax.fori_loop(
            0, num_steps, lambda i, state: integrator(state, step_size), state
        )
        return new_state

    return propose


#
# Integrators
#


def velocity_verlet(potential_fn: Callable, kinetic_energy_fn: Callable) -> Integrator:
    """The velocity Verlet integrator, a one-stage second order symplectic integrator [1]_.

    The velocity verlet is a palindromic integrator of the form (b1, a1, b1).
    The velocity verlet has a normalized (by the number of force evaluations
    per time step) stability interval of 2.

    Note
    ----

    Micro-benchmarks on the harmonic oscillator show that pre-computing the
    gradient of the log-probability density function and the kinetic energy
    yields a 10% performance improvement.

    (TODO) reproduce on a more complex potential.

    References
    ----------
    .. [1]: Bou-Rabee, Nawaf, and Jesús Marıa Sanz-Serna. "Geometric
            integrators and the Hamiltonian Monte Carlo method." Acta Numerica 27
            (2018): 113-206.
    """

    b1 = 0.5
    a1 = 1.0

    potential_grad_fn = jax.jit(jax.grad(potential_fn))
    kinetic_energy_grad_fn = jax.jit(jax.grad(kinetic_energy_fn))

    @partial(jax.jit, static_argnums=(1,))
    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, log_prob_grad = state

        momentum = momentum - b1 * step_size * log_prob_grad

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = position + a1 * step_size * kinetic_grad

        log_prob_grad = potential_grad_fn(position)
        momentum = momentum - b1 * step_size * log_prob_grad

        return IntegratorState(position, momentum, log_prob_grad)

    return one_step


def mclachlan_integrator(
    potential_fn: Callable, kinetic_energy_fn: Callable
) -> Integrator:
    """Two-stage palindromic symplectic integrator derived in [1]_

    The integrator is of the form (b1, a1, b2, a1, b1). The choice of the parameters
    determine both the bound on the integration error and the stability of the
    method with respect to the value of `step_size`. The values used here are
    the ones derived in [2]_; note that [1]_ is more focused on stability
    and derives different values.

    References
    ----------
    .. [1]: Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
            integrators for the Hybrid Monte Carlo method." SIAM Journal on Scientific
            Computing 36.4 (2014): A1556-A1580.
    .. [2]: McLachlan, Robert I. "On the numerical integration of ordinary
            differential equations by symmetric composition methods." SIAM Journal on
            Scientific Computing 16.1 (1995): 151-168.
    """

    b1 = 0.1932
    a1 = 0.5
    b2 = 1 - 2 * b1

    potential_grad_fn = jax.jit(jax.grad(potential_fn))
    kinetic_energy_grad_fn = jax.jit(jax.grad(kinetic_energy_fn))

    @partial(jax.jit, static_argnums=(1,))
    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, log_prob_grad = state

        momentum = momentum - b1 * step_size * log_prob_grad

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = position + a1 * step_size * kinetic_grad

        log_prob_grad = potential_grad_fn(position)
        momentum = momentum - b2 * step_size * log_prob_grad

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = position + a1 * step_size * kinetic_grad

        log_prob_grad = potential_grad_fn(position)
        momentum = momentum - b1 * step_size * log_prob_grad

        return IntegratorState(position, momentum, log_prob_grad)

    return one_step


def yoshida_integrator(
    potential_fn: Callable, kinetic_energy_fn: Callable,
) -> Integrator:
    """Three stages palindromic symplectic integrator derived in [1]_

    The integrator is of the form (b1, a1, b2, a2, b2, a1, b1). The choice of
    the parameters determine both the bound on the integration error and the
    stability of the method with respect to the value of `step_size`. The
    values used here are the ones derived in [1]_ which guarantees a stability
    interval length approximately equal to 4.67.

    References
    ----------
    .. [1]: Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
            integrators for the Hybrid Monte Carlo method." SIAM Journal on Scientific
            Computing 36.4 (2014): A1556-A1580.
    """

    b1 = 0.11888010966548
    a1 = 0.29619504261126
    b2 = 0.5 - b1
    a2 = 1 - 2 * a1

    potential_grad_fn = jax.jit(jax.grad(potential_fn))
    kinetic_energy_grad_fn = jax.jit(jax.grad(kinetic_energy_fn))

    @partial(jax.jit, static_argnums=(1,))
    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, log_prob_grad = state

        momentum = momentum - b1 * step_size * log_prob_grad

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = position + a1 * step_size * kinetic_grad

        log_prob_grad = potential_grad_fn(position)
        momentum = momentum - b2 * step_size * log_prob_grad

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = position + a2 * step_size * kinetic_grad

        log_prob_grad = potential_grad_fn(position)
        momentum = momentum - b2 * step_size * log_prob_grad

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = position + a1 * step_size * kinetic_grad

        log_prob_grad = potential_grad_fn(position)
        momentum = momentum - b1 * step_size * log_prob_grad

        return IntegratorState(position, momentum, log_prob_grad)

    return one_step


def four_stages_integrator(
    potential_fn: Callable, kinetic_energy_fn: Callable,
) -> Integrator:
    """Four stages palindromic symplectic integrator derived in [1]_

    The integrator is of the form (b1, a1, b2, a3, b2, a2, b2, a1, b1). The choice of
    the parameters determine both the bound on the integration error and the
    stability of the method with respect to the value of `step_size`. The
    values used here are the ones derived in [1]_ which guarantees a stability
    interval length approximately equal to 5.35.

    References
    ----------
    .. [1]: Blanes, Sergio, Fernando Casas, and J. M. Sanz-Serna. "Numerical
            integrators for the Hybrid Monte Carlo method." SIAM Journal on Scientific
            Computing 36.4 (2014): A1556-A1580.
    """

    b1 = 0.071353913450279725904
    b2 = 0.268548791161230105820
    a1 = 0.1916678
    a2 = 0.5 - a1
    b3 = 1 - 2 * b1 - 2 * b2

    potential_grad_fn = jax.jit(jax.grad(potential_fn))
    kinetic_energy_grad_fn = jax.jit(jax.grad(kinetic_energy_fn))

    @partial(jax.jit, static_argnums=(1,))
    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, log_prob_grad = state

        momentum = momentum - b1 * step_size * log_prob_grad

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = position + a1 * step_size * kinetic_grad

        log_prob_grad = potential_grad_fn(position)
        momentum = momentum - b2 * step_size * log_prob_grad

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = position + a2 * step_size * kinetic_grad

        log_prob_grad = potential_grad_fn(position)
        momentum = momentum - b3 * step_size * log_prob_grad

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = position + a2 * step_size * kinetic_grad

        log_prob_grad = potential_grad_fn(position)
        momentum = momentum - b2 * step_size * log_prob_grad

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = position + a1 * step_size * kinetic_grad

        log_prob_grad = potential_grad_fn(position)
        momentum = momentum - b1 * step_size * log_prob_grad

        return IntegratorState(position, momentum, log_prob_grad)

    return one_step
