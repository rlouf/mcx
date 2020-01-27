# encoding: utf-8
"""
Sampling methods using the Random Walk Metropolis algorithm.
"""
from typing import Callable, Generator
from functools import partial

import jax
import jax.numpy as np
import numpy as onp


def hmc_generator(
    rng_key: jax.random.PRNGKey,
    logpdf: Callable,
    integrator: Callable,
    initial_position: jax.numpy.ndarray,
    initial_step_size: float = 0.1,
    path_length: int = 1,
) -> Generator[onp.ndarray, None, None]:
    """Returns one sample at a time for an arbitrary numbre of chains using the
    Hamiltonian Monte Carlo algorithm.

    Parameters
    ----------
    rng_key: jax.random.PRNGKey
        The key for Jax's pseudo random number generator.
    logpdf: function
        The target being sampled. Evaluated at the current position it
        returns the model's log probability of the current position and
        the gradient of the log probability at this position.
    integrator: function
        The function used to integrate the hamiltonian dymamics.
    initial_position: np.ndarray (n_features, n_chains)
        The initial state of the chain. For a unique starting point a
        simple vector that contain the initial position from which to
        sample. For several starting point, a matrix.
    initial_step_size: float
        The initial size of the integration steps.
    path_length: int
        The length of the integration path.

    Yields
    ------
    jax.numpy.array (1 ,n_vars)
        The samples
    """
    n_chains = initial_position.shape[0]
    position = initial_position
    log_prob, log_prob_grad = logpdf(initial_position)  # compiled, autograd
    yield onp.as_array(position)

    step_size = initial_step_size
    while True:
        rng_key, sample_key = jax.random.split(rng_key)
        chains_keys = jax.random.split(sample_key, n_chains)
        position, log_prob, logprob_grad = jax.vmap(
            hmc_kernel, in_axes=(0, None, None, None, None, 0, 0, 0), out_axes=(0,)
        )(
            chains_keys,
            logpdf,
            integrator,
            step_size,
            path_length,
            position,
            log_prob,
            log_prob_grad,
        )
        yield onp.as_array(position)


def hmc_sampler(
    rng_key: jax.random.PRNGKey,
    logpdf: Callable,
    integrator: Callable,
    initial_position: jax.numpy.ndarray,
    n_samples: int = 1000,
    initial_step_size: float = 0.1,
    path_length: int = 1,
) -> onp.ndarray:

    n_chains = initial_position.shape[0]
    rng_keys = jax.random.split(rng_key, n_chains)
    position = jax.vmap(hmc_single_chain, in_axes=(0, None, None, None, 0), out_axes=0)(
        rng_keys, n_samples, logpdf, integrator, initial_position
    )
    return onp.as_array(position)


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


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def hmc_kernel(rng_key, logpdf, integrator, path_length, step_size, state):
    """

    Returns
    -------
    tuple
        The new position, log-probability at the position and the gradient
        of the log-pobability at this position.
    float
        The acceptance probability.
    """
    momentum_key, uniform_key = jax.random.split(rng_key)

    position, log_prob, log_prob_grad = state
    n_features = position.shape[0]

    # Hamiltonian proposal
    # Implicitly a diagonal, constant, mass matrix
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


@partial(jax.jit, static_argnums=(4,))
def leapfrog_integrator(position, momentum, potential, potential_grad, path_length, step_size):
    """Second order symplectic integrator that uses the leapfrog algorithm
    """
    position, momentum = np.copy(position), np.copy(momentum)
    momentum -= step_size * potential_grad / 2  # half step
    for _ in np.arange(np.round(path_length / step_size) - 1):
        position = position + step_size * momentum  # whole step
        potential_value, potential_grad = potential(position)
        momentum = momentum - step_size * potential_grad  # whole step
    position = position + step_size * momentum  # whole step
    potential_value, potential_grad = potential(position)
    momentum = momentum - step_size * potential_grad / 2  # half step

    return position, -momentum, potential_value, potential_grad
