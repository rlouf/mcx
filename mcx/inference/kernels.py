"""Sampling kernels.

.. note:
    This file is a "flat zone": positions and logprobs are 1 dimensional
    arrays. Raveling and unraveling logic must happen outside.
"""
from functools import partial
from typing import Callable, NamedTuple, Tuple

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
    rng_key: jax.random.RNGKey,
    logpdf: Callable,
    integrator: Callable,
    momentum_generator: Callable,
    kinetic_energy: Callable,
    path_length_generator: Callable,
    step_size: float,
):
    """Hamiltonian Monte Carlo transition kernel.

    Provides three functions to propose one step using Hamiltonian dynamics,
    accept or reject this step using the Metropolis-Hastings algorithm, and
    finally a kernel function that provides both.

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
    path_length_generator:
        A function that returns a new value for the path length when called.
    step_size:
        The step size to use when integrating the trajectory.
    state:
        The current state of the chain: position, log-probability and gradient
        of the log-probability.

    """

    def step(rng_key, state: HMCState) -> HMCState:
        """Moves the chain by one step using the Hamiltonian dynamics.
        """
        key_momentum, key_path = jax.random.split(rng_key)
        position, log_prob, log_prob_grad, _ = state

        momentum = momentum_generator(key_momentum)
        position, momentum, log_prob, log_prob_grad = integrator(
            position,
            momentum,
            log_prob_grad,
            log_prob,
            path_length_generator(key_path),
            step_size=step_size,
        )
        energy = log_prob + kinetic_energy(momentum)

        return HMCState(position, log_prob, log_prob_grad, energy)

    def accept(
        rng_key: jax.random.PRNGKey, state: HMCState, previous_state: HMCState
    ) -> HMCState:

        log_uniform = np.log(jax.random.uniform(rng_key))
        do_accept = log_uniform < state.energy - previous_state.energy
        if do_accept:
            return state

        return previous_state

    def kernel(rng_key, state: HMCState):
        step_key, accept_key = jax.random.split(rng_key)
        proposal_state = step(step_key, state)
        final_state = accept(accept_key, proposal_state, state)
        return final_state

    return kernel, step, accept


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
