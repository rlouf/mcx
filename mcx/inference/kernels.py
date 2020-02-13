"""Sampling kernels.

.. note:
    This file is a "flat zone": positions and logprobs are 1 dimensional
    arrays. Raveling and unraveling logic must happen outside.
"""
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as np
import jax.numpy.DeviceArray as Array

__all__ = ["hmc_kernel", "rwm_kernel"]


class HMCState(NamedTuple):
    position: Array
    log_prob: float
    log_prob_grad: float
    energy: float


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def hmc_kernel(
    logpdf: Callable,
    integrator: Callable,
    momentum_generator: Callable,
    kinetic_energy: Callable,
):
    """Hamiltonian Monte Carlo transition kernel factory.

    Returns a kernel based on the hamiltonian monte carlo algorithm, a map
    between different states of the phase space.

    Arguments
    ---------
    rng_key:
       The pseudo-random number generator key used to generate random numbers.
    logpdf:
        The logpdf of the model whose posterior we want to sample. Returns the
        log probability and gradient when evaluated at a position.
    integrator:
        The function used to integrate the equations of motion.
    momentum_generator:
        A function that returns a new value for the momentum when called.
    kinetic_energy:
        A function that computes the trajectory's kinetic energy.
    state:
        The current state of the chain: position, log-probability and gradient
        of the log-probability.

    """

    def kernel(rng_key, state: HMCState) -> HMCState:
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
        The next state of the chain.
        """
        key_momentum, key_integrator, key_uniform = jax.random.split(rng_key, 3)
        position, log_prob, log_prob_grad, energy = state

        momentum = momentum_generator(key_momentum)
        position, momentum, log_prob, log_prob_grad = integrator(
            key_integrator, position, momentum, log_prob_grad, log_prob,
        )
        new_energy = log_prob + kinetic_energy(momentum)
        new_state = HMCState(position, log_prob, log_prob_grad, energy)

        log_uniform = np.log(jax.random.uniform(key_uniform))
        do_accept = log_uniform < energy - new_energy
        if do_accept:
            return new_state

        return state

    return kernel


class RWMState(NamedTuple):
    position: Array
    log_prob: float


@partial(jax.jit, static_argnums=(1, 2))
def rwm_kernel(
    rng_key: jax.random.PRNGKey, logpdf: Callable, proposal: Callable, state: RWMState
) -> RWMState:
    """Random Walk Metropolis transition kernel.

    Moves the chain by one step using the Random Walk Metropolis algorithm.

    Arguments
    ---------
    rng_key: jax.random.PRNGKey
        Key for the pseudo random number generator.
    logpdf: function
        Returns the log-probability of the model given a position.
    proposal: function
        Returns a move proposal.
    state: RWMState
        The current state of the markov chain.

    Returns
    -------
    RMWState
        The new state of the markov chain.
    """
    key_move, key_uniform = jax.random.split(rng_key)

    position, log_prob = state

    move_proposal = proposal(key_move)
    proposal = position + move_proposal
    proposal_log_prob = logpdf(proposal)

    log_uniform = np.log(jax.random.uniform(key_uniform))
    do_accept = log_uniform < proposal_log_prob - log_prob

    position = np.where(do_accept, proposal, position)
    log_prob = np.where(do_accept, proposal_log_prob, log_prob)
    return RWMState(position, log_prob)
