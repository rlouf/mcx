"""Sampling kernels.
"""
from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as np

from mcx.inference.integrators import ProposalState, ProposalInfo, Proposer
from mcx.inference.metrics import KineticEnergy, MomentumGenerator


__all__ = ["HMCState", "HMCInfo", "hmc_init", "hmc_kernel", "rwm_kernel"]


# ----------------------------------------
#      == Hamiltonian Monte Carlo ==
# ----------------------------------------


class HMCState(NamedTuple):
    """Describes the state of the HMC algorithm."""

    position: np.DeviceArray
    potential_energy: float
    potential_energy_grad: float


class HMCInfo(NamedTuple):
    """Additional information on the current HMC step that can be useful for
    debugging or diagnostics.
    """

    proposed_state: HMCState
    acceptance_probability: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: ProposalState
    proposal_info: ProposalInfo


@partial(jax.jit, static_argnums=(1,))
def hmc_init(position: np.DeviceArray, potential_value_and_grad: Callable) -> HMCState:
    """Compute the initial state of the HMC algorithm from the initial position
    and the log-likelihood.
    """
    potential_energy, potential_energy_grad = potential_value_and_grad(position)
    return HMCState(position, potential_energy, potential_energy_grad)


def hmc_kernel(
    proposal_generator: Proposer,
    momentum_generator: MomentumGenerator,
    kinetic_energy: KineticEnergy,
    potential: Callable,
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
    energy). While the hamiltonian dynamics is conservative, numerical
    integration can introduce some discrepancy; we perform a Metropolis
    acceptance test to compensate for integration errors after having flipped
    the new state's momentum to make the transition reversible.

    I encourage anyone interested in the theoretical underpinning of the method
    to read Michael Betancourts' excellent introduction [Betancourt2018]_ and his
    more technical paper [Betancourt2017]_ on the geometric foundations of the method.

    This implementation is very general and should accomodate most variations
    on the method.

    Parameters
    ----------
    proposal_generator:
        The function used to propose a new state for the chain. For vanilla HMC this
        function integrates the trajectory over many steps, but gets more involved
        with other algorithms such as empirical and dynamical HMC.
    momentum_generator:
        A function that returns a new value for the momentum when called.
    kinetic_energy:
        A function that computes the current state's kinetic energy.
    potential:
        The potential function that is being explored, equal to minus the likelihood.
    divergence_threshold:
        The maximum difference in energy between the initial and final state
        after which we consider the transition to be divergent.

    Returns
    -------
    A kernel that moves the chain by one step when called.

    References
    ----------
    .. [Duane1987]: Duane, Simon, et al. "Hybrid monte carlo." Physics letters B
                    195.2 (1987): 216-222.
    .. [Neal1994]: Neal, Radford M. "An improved acceptance procedure for the
                   hybrid Monte Carlo algorithm." Journal of Computational Physics 111.1
                   (1994): 194-203.
    .. [Betancourt2017]: Betancourt, Michael, et al. "The geometric foundations
                          of hamiltonian monte carlo." Bernoulli 23.4A (2017): 2257-2298.
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

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random numbers.
        state:
            The current state of the chain: position, log-probability and gradient
            of the log-probability.

        Returns
        -------
        The next state of the chain and additional information about the current step.
        """
        key_momentum, key_integrator, key_accept = jax.random.split(rng_key, 3)

        position, potential_energy, potential_energy_grad = state
        momentum = momentum_generator(key_momentum)
        energy = potential_energy + kinetic_energy(momentum)

        proposal, proposal_info = proposal_generator(
            key_integrator, ProposalState(position, momentum, potential_energy_grad)
        )
        new_position, new_momentum, new_potential_energy_grad = proposal

        flipped_momentum = -1.0 * new_momentum  # to make the transition reversible
        new_potential_energy = potential(new_position)
        new_energy = new_potential_energy + kinetic_energy(flipped_momentum)
        new_state = HMCState(
            new_position, new_potential_energy, new_potential_energy_grad
        )

        delta_energy = energy - new_energy
        delta_energy = np.where(np.isnan(delta_energy), -np.inf, delta_energy)
        is_divergent = np.abs(delta_energy) > divergence_threshold

        p_accept = np.clip(np.exp(delta_energy), a_max=1)
        do_accept = jax.random.bernoulli(key_accept, p_accept)
        accept_state = (
            new_state,
            HMCInfo(
                new_state,
                p_accept,
                True,
                is_divergent,
                new_energy,
                proposal,
                proposal_info,
            ),
        )
        reject_state = (
            state,
            HMCInfo(
                new_state,
                p_accept,
                False,
                is_divergent,
                energy,
                proposal,
                proposal_info,
            ),
        )
        return jax.lax.cond(
            do_accept,
            accept_state,
            lambda state: state,
            reject_state,
            lambda state: state,
        )

    return kernel


# --------------------------------------
#      == Random Walk Metropolis ==
# --------------------------------------


class RWMState(NamedTuple):
    """Describes the state of the Random Walk Metropolis algorithm."""

    position: np.DeviceArray
    log_prob: float


class RWMInfo(NamedTuple):
    """Additional information on the current Random Walk Metropolis step that
    can be useful for debugging or diagnostics.
    """

    proposed_state: RWMState
    acceptance_probability: float
    is_accepted: bool


def rwm_kernel(logpdf: Callable, proposal_generator: Callable) -> Callable:
    """Creates a Random Walk Metropolis transition kernel.

    Parameters
    ----------
    logpdf:
        Returns the log-probability of the model given a position.
    proposal_generator:
        The function used to propose a new state for the chain.

    Returns
    -------
    A kernel that moves the chain by one step when called.
    """

    @jax.jit
    def kernel(
        rng_key: jax.random.PRNGKey, state: RWMState
    ) -> Tuple[RWMState, RWMInfo]:
        """Moves the chain by one step using the Random Walk Metropolis algorithm.

        Parameters
        ----------
        rng_key:
           The pseudo-random number generator key used to generate random numbers.
        state:
            The current state of the chain: position, log-probability and gradient
            of the log-probability.

        Returns
        -------
        The next state of the chain and additional information about the current step.
        """
        key_move, key_accept = jax.random.split(rng_key)

        position, log_prob = state

        move_proposal = proposal_generator(key_move)
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
