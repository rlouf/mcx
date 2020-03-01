"""Sampling kernels.

.. note:
    This file is a "flat zone": positions and logprobs are 1 dimensional
    arrays. Raveling and unraveling logic must happen outside.
"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as np
from jax.numpy import DeviceArray as Array

from mcx.inference.integrators import IntegratorState
from mcx.inference.metrics import MomentumGenerator, KineticEnergy


__all__ = ["HMCState", "HMCInfo", "hmc_init", "hmc_kernel", "rwm_kernel"]


class HMCState(NamedTuple):
    """Describes the state of the HMC kernel.
    """

    position: Array
    log_prob: float
    log_prob_grad: float


class HMCInfo(NamedTuple):
    """Additional information on the current HMC step.
    """

    proposed_state: HMCState
    acceptance_probability: float
    is_accepted: bool
    is_divergent: bool
    integrator_state: IntegratorState


def hmc_init(position: Array, logpdf: Callable) -> HMCState:
    log_prob, log_prob_grad = jax.value_and_grad(logpdf)(position)
    return HMCState(position, log_prob, log_prob_grad)


def hmc_kernel(
    integrator: Callable[[jax.random.PRNGKey, IntegratorState], IntegratorState],
    momentum_generator: MomentumGenerator,
    kinetic_energy: KineticEnergy,
    divergence_threshold: float = 1000.0,
) -> Callable:
    """Creates a Hamiltonian Monte Carlo transition kernel.

    Hamiltonian Monte Carlo (HMC) is known to yield effective Markov
    transitions and has been a major empirical success, leading to an extensive
    use in probabilistic programming languages and libraries [Duane1987,
    Neal1994, Betancourt2018]_.

    HMC works by augmenting the state space in which the chain evolves with an
    auxiliary momentum :math:`p`. At each step of the chain we draw a momentum
    value from the `momentum_generator` function. We then use Hamilton's
    equations [HamiltonEq]_ to push the state forward; we then compute the new
    state's energy using the `kinetic_energy` function and `logpdf` (potential
    energy). While the hamiltonian dynamics is conservative, numerically
    integration can introduce some discrepancy; we perform a metropolis
    acceptance test to compensate for integration errors after having flipped
    the new state's momentum to make the transition reversible.

    I encourage anyone interested in the theoretical underpinning of the method
    to read Michael Betancourts' excellent introduction [Betancourt2018]_.

    This implementation is very general and should accomodate most variations
    of the method.

    Arguments
    ---------
    integrator:
        The function used to integrate the equations of motion.
    momentum_generator:
        A function that returns a new value for the momentum when called.
    kinetic_energy:
        A function that computes the trajectory's kinetic energy.
    divergence_threshold:
        The maximum difference in energy between the initial and final state
        after which we consider the transition to be divergent.

    Returns
    -------
    A kernel that moves the chain by one step.

    References
    ----------
    .. [Duane1987]: Duane, Simon, et al. "Hybrid monte carlo." Physics letters B
                    195.2 (1987): 216-222.
    .. [Neal1994]: Neal, Radford M. "An improved acceptance procedure for the
                   hybrid Monte Carlo algorithm." Journal of Computational Physics 111.1
                   (1994): 194-203.
    .. [Betancourt2018]: Betancourt, Michael. "A conceptual introduction to
                         Hamiltonian Monte Carlo." arXiv preprint arXiv:1701.02434 (2018).
    .. [HamiltonEq]: "Hamiltonian Mechanics", Wikipedia.
                     https://en.wikipedia.org/wiki/Hamiltonian_mechanics#Deriving_Hamilton's_equations
    """

    @jax.jit
    def kernel(
        rng_key: jax.random.PRNGKey, state: HMCState
    ) -> Tuple[HMCState, HMCInfo]:
        """Moves the chain by one step using the Hamiltonian dynamics.

        Arguments
        ---------
        rng_key:
           The pseudo-random number generator key used to generate random numbers.
        state:
            The current state of the chain: position, log-probability and gradient
            of the log-probability.

        Returns
        -------
        Next state of the chain and information about the current step.
        """
        key_momentum, key_integrator, key_accept = jax.random.split(rng_key, 3)

        position, log_prob, log_prob_grad = state
        momentum = momentum_generator(key_momentum)
        integrator_state = integrator(
            key_integrator,
            IntegratorState(position, momentum, log_prob, log_prob_grad),
        )
        new_position, new_momentum, new_log_prob, new_log_prob_grad = (
            integrator_state.position,
            integrator_state.momentum,
            integrator_state.log_prob,
            integrator_state.log_prob_grad,
        )
        new_state = HMCState(new_position, new_log_prob, new_log_prob_grad)

        flipped_momentum = -1.0 * new_momentum  # to make the transition reversible
        energy = log_prob + kinetic_energy(momentum)
        new_energy = new_log_prob + kinetic_energy(flipped_momentum)

        delta_energy = energy - new_energy
        delta_energy = np.where(np.isnan(delta_energy), -np.inf, delta_energy)
        is_divergent = np.abs(delta_energy) > divergence_threshold

        p_accept = np.clip(np.exp(delta_energy), a_max=1)
        do_accept = jax.random.bernoulli(key_accept, p_accept)
        accept_state = (
            new_state,
            HMCInfo(new_state, p_accept, True, is_divergent, integrator_state),
        )
        reject_state = (
            state,
            HMCInfo(new_state, p_accept, False, is_divergent, integrator_state),
        )
        return jax.lax.cond(
            do_accept,
            accept_state,
            lambda state: state,
            reject_state,
            lambda state: state,
        )

    return kernel


#
# Random Walk Metropolis
#


class RWMState(NamedTuple):
    """Describes the state of the Random Walk Metropolis chain.
    """

    position: Array
    log_prob: float


class RWMInfo(NamedTuple):
    """Additional information on the current Random Walk Metropolis step.
    """

    proposed_state: RWMState
    acceptance_probability: float
    is_accepted: bool


def rwm_kernel(logpdf: Callable, proposal_fn: Callable) -> Callable:
    """Random Walk Metropolis transition kernel.

    Moves the chain by one step using the Random Walk Metropolis algorithm.

    Arguments
    ---------
    logpdf: function
        Returns the log-probability of the model given a position.
    proposal_fn: function
        Returns a move proposal.

    Returns
    -------
    A kernel that moves the chain by one step.
    """

    @jax.jit
    def kernel(
        rng_key: jax.random.PRNGKey, state: RWMState
    ) -> Tuple[RWMState, RWMInfo]:
        key_move, key_accept = jax.random.split(rng_key)

        position, log_prob = state

        move_proposal = proposal_fn(key_move)
        new_position = position + move_proposal
        new_log_prob = logpdf(new_position)
        new_state = RWMState(new_position, new_log_prob)

        delta = new_log_prob - log_prob
        delta = np.where(np.isnan(delta), -np.inf, delta)
        p_accept = np.clip(np.exp(delta), a_max=1)

        do_accept = jax.random.bernoulli(key_accept, p_accept)
        accept_state = (new_state, RWMInfo(new_state, p_accept, True))
        reject_state = (state, RWMInfo(new_state, p_accept, False))
        return np.where(do_accept, accept_state, reject_state)

    return kernel
