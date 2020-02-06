# encoding: utf-8
"""
Sampling methods using the Random Walk Metropolis algorithm.
"""
from functools import partial
from typing import Callable, Tuple, Union

import jax
import jax.numpy as np
import numpy

Array = Union[numpy.ndarray, jax.numpy.DeviceArray]


@partial(jax.jit, static_argnums=(1, 2, 3))
def hmc_single_chain(
    rng_key, n_samples, logpdf, integrator, path_length, step_size, initial_position
) -> jax.numpy.ndarray:
    def mh_update(_, state):
        key, state = state
        _, key = jax.random.split(key)
        position, log_prob, log_prob_grad = hmc_kernel(
            key, logpdf, integrator, path_length, step_size, state
        )
        return (key, position, log_prob, log_prob_grad)

    position = initial_position
    log_prob, log_prob_grad = logpdf(initial_position)
    _, position, _, _ = jax.lax.fori_loop(
        0, n_samples, mh_update, (rng_key, position, log_prob, log_prob_grad)
    )
    return position


HMCState = Tuple[Array, Array, Array]


@partial(jax.jit, static_argnums=(1, 2))
def hmc_kernel(
    rng_key: jax.random.PRNGKey,
    logpdf: Callable,
    integrator: Callable,
    mass_matrix: Array,
    path_length: float,
    step_size: float,
    state: HMCState,
) -> HMCState:
    """Hamiltonian Monte Carlo transition kernel.

    Moves the chains by one step using the Hamiltonian Monte Carlo algorithm.
    The kernel implementation is made as general as possible to ease re-use by
    different Monte Carlo algorithms; adaptive schemes are left for algorithm to
    use.

    Arguments
    ---------
    rng_key:
       The pseudo-random number generator key used to generate random numbers.
    logpdf:
        The logpdf of the model whose posterior we want to sample. Returns the
        log probability and gradient when evaluated at a position.
    integrator:
        The function used to integrate the equations of motion.
    mass_matrix:
        The mass matrix of the euclidean metric.
    path_length:
        The current number of integration steps.
    step_size:
        The current size of integration steps.
    state:
        The current state of the chain: position, log-probability and gradient
        of the log-probability.


    Returns
    -------
    HMCState
        The new position, log-probability at the position and the gradient
        of the log-pobability at this position.
    """
    momentum_key, uniform_key = jax.random.split(rng_key)

    position, log_prob, log_prob_grad = state
    n_features = position.shape[0]

    # Hamiltonian proposal
    # Implicitly a diagonal, constant, mass matrix
    # Add mass matrix here
    momentum = jax.random.normal(momentum_key, (1, n_features))
    proposal, p_new, proposal_log_prob, proposal_log_prob_grad = integrator(
        position,
        momentum,
        log_prob_grad,
        log_prob,
        path_length=2 * jax.random.uniform(uniform_key) * path_length,
        step_size=step_size,
    )

    # Metropolis Hastings acceptance step
    start_log_p = np.sum(momentum.logpdf(momentum)) - log_prob
    new_log_p = np.sum(momentum.logpdf(p_new)) - proposal_log_prob
    log_uniform = np.log(jax.random.uniform(uniform_key))
    do_accept = log_uniform < new_log_p - start_log_p
    if do_accept:
        position = proposal
        log_prob = proposal_log_prob
        log_prob_grad = proposal_log_prob_grad

    state = (position, log_prob, log_prob_grad)

    return state
