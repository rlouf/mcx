"""Warming up the chain.

.. note:
    This is a "flat zone": all positions are 1D array.
"""
from functools import partial
from typing import Callable, Tuple

import jax
from jax import numpy as np

from mcx.inference.adaptive import (
    dual_averaging,
    find_reasonable_step_size,
    mass_matrix_adaptation,
    MassMatrixAdaptationState,
    longest_batch_before_turn,
)
from mcx.inference.dynamics import gaussian_euclidean_metric
from mcx.inference.integrators import leapfrog
from mcx.inference.kernels import HMCState, hmc_kernel


def hmc_warmup(
    rng_key: jax.random.PRNGKey,
    logpdf: Callable,
    initial_state: HMCState,
    inital_step_size: float,
    path_length: float,
    num_steps: int,
    diagonal_mass_matrix=True,
) -> Tuple[HMCState, float, MassMatrixAdaptationState]:
    """ Warmup scheme for sampling procedures based on euclidean manifold HMC.

    Separation between sampling and warmup ensures better modularity; a modification
    in the warmup procedure should not affect the sampling implementation.

    Returns
    -------
    Tuple
        The current state of the chain and the warmed-up kernel.
    """

    n_dims = np.shape(initial_state.position)[-1]  # `position` is a 1D array

    # Initialize the mass matrix adaptation
    mm_init, mm_update, mm_final = mass_matrix_adaptation(diagonal_mass_matrix)
    mm_state = mm_init(n_dims)

    # Initialize the HMC transition kernel
    momentum_generator, kinetic_energy = gaussian_euclidean_metric(
        mm_state.mass_matrix_sqrt, mm_state.inverse_mass_matrix
    )
    hmc_partial_kernel = jax.partial(
        hmc_kernel,
        logpdf,
        leapfrog_integrator,
        momentum_generator,
        kinetic_energy,
        lambda x: path_length,
    )

    # Initialize the dual averaging
    step_size = find_reasonable_step_size(
        rng_key, hmc_partial_kernel, initial_state, inital_step_size,
    )
    da_init, da_update = dual_averaging()
    da_state = da_init(step_size)

    # Get warmup schedule
    schedule = warmup_schedule(num_steps)

    state = initial_state
    for i, window in enumerate(schedule):

        for step in range(window):
            accept = lambda x, y, z: x
            proposal_state = hmc_partial_kernel(rng_key, step_size)
            state = accept(rng_key, proposal_state, state)
            if i != 0 and i != len(schedule) - 1:
                mm_state = mm_update(mm_state, state.position)

        if i == 0:
            da_state = da_update(state.p_accept, da_state)
            step_size = np.exp(da_state.log_step_size)
        elif i == len(schedule) - 1:
            da_state = da_update(state.p_accept, da_state)
            step_size = np.exp(da_state.log_step_size_avg)
        else:
            inverse_mass_matrix, mass_matrix_sqrt = mm_final(mm_state)
            momentum_generator, kinetic_energy = gaussian_euclidean_metric(
                mm_state.mass_matrix_sqrt, mm_state.inverse_mass_matrix
            )
            hmc_partial_kernel = jax.partial(
                hmc_kernel,
                logpdf,
                leapfrog_integrator,
                momentum_generator,
                kinetic_energy,
                lambda x: path_length,
            )

    kernel = hmc_partial_kernel(step_size)

    return state, step_size, mm_state


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7))
def ehmc_warmup(
    rng_key: jax.random.PRNGKey,
    logpdf: Callable,
    initial_state: HMCState,
    inital_step_size: float,
    path_length: float,
    num_hmc_warmup_steps: int,
    num_longest_batch: int,
    diagonal_mass_matrix=True,
):
    """Warmup scheme for empirical Hamiltonian Monte Carlo.

    We first run the standard HMC warmup to adapt the step size and mass
    matrix. Once this first step is performed, we build a list of longest
    batches (path lengths) from which we will draw path lengths for the
    integration step during inference.

    For that purpose, we build a custom integrator that implements algorithm
    2. in [1]_.

    References
    ----------
    .. [1]: Wu, Changye, Julien Stoehr, and Christian P. Robert. "Faster
            Hamiltonian Monte Carlo by learning leapfrog scale." arXiv preprint
            arXiv:1810.04449 (2018).
    """
    hmc_key, ehmc_key = jax.random.split(rng_key)

    hmc_warmup_state, step_size, mm_state = hmc_warmup(
        hmc_key,
        logpdf,
        initial_state,
        inital_step_size,
        path_length,
        num_hmc_warmup_steps,
        diagonal_mass_matrix,
    )

    # Build the warmup kernel

    momentum_generator, kinetic_energy = gaussian_euclidean_metric(
        mm_state.mass_matrix_sqrt, mm_state.inverse_mass_matrix
    )
    leapfrog_step = leapfrog(logpdf)
    longest_batch_step = longest_batch_before_turn(leapfrog_step)

    def longest_batch_integrator(rng_key, integrator_state):
        position, momentum, log_prob, log_prob_grad, batch_length = longest_batch_step(
            position, momentum, step_size, path_length
        )
        if batch_length < path_length:
            position, momentum, log_prob, log_prob_grad = leapfrog_step(
                position,
                momentum,
                log_prob,
                log_prob_grad,
                step_size,
                path_length - batch_length,
            )
        return momentum, position, log_prob, log_prob_grad, batch_length

    ehmc_warmup_kernel = hmc_kernel(
        longest_batch_integrator, momentum_generator, kinetic_energy
    )

    # Run the kernel and return an array of longest batch lengths

    def warmup_update(state, key):
        hmc_state, _ = state
        new_state, new_info = ehmc_warmup_kernel(key, hmc_state)
        _, _, _, _, batch_length = new_info.integrator_step
        return (new_state, new_info), batch_length

    keys = jax.random.split(ehmc_key, num_longest_batch)
    state, batch_lengths = jax.lax.scan(warmup_update, (hmc_warmup_state, None), keys,)
    hmc_warmup_state, _ = state

    return hmc_warmup_state, batch_lengths, step_size, mm_state


def warmup_schedule(num_steps, initial_buffer=75, first_window=25, final_buffer=50):
    """Returns an adaptation warmup schedule.

    The schedule below is intended to be as close as possible to Stan's _[1].
    The warmup period is split into three stages:

    1. An initial fast interval to reach the typical set.
    2. "Slow" parameters that require global information (typically covariance)
       are estimated in a series of expanding windows with no memory.
    3. Fast parameters are learned after the adaptation of the slow ones.

    See _[1] for a more detailed explanation.

    Parameters
    ----------
    num_warmup: int
        The number of warmup steps to perform.
    initial_buffer: int
        The width of the initial fast adaptation interval.
    first_window: int
        The width of the first slow adaptation interval. There are 5 such
        intervals; the width of a window interval is twice the size of the
        preceding.
    final_buffer: int
        The width of the final fast adaptation interval.

    References
    ----------
    .. [1]: Stan Reference Manual v2.22
            Section 15.2 "HMC Algorithm"
    """
    schedule = []

    # Handle the situations where the numbrer of warmup steps is smaller than
    # the sum of the buffers' widths
    if num_steps < 20:
        schedule.append((0, num_steps - 1))
        return schedule

    if initial_buffer + first_window + final_buffer > num_steps:
        initial_buffer = int(0.15 * num_steps)
        final_buffer = int(0.1 * num_steps)
        first_window = num_steps - initial_buffer - final_buffer

    # First stage: adaptation of fast parameters
    schedule.append((0, initial_buffer - 1))

    # Second stage: adaptation of slow parameters
    final_buffer_start = num_steps - final_buffer

    next_size = first_window
    next_start = initial_buffer
    while next_start < final_buffer_start:
        start, size = next_size, next_start
        if 3 * size <= final_buffer_start - start:
            next_size = 2 * size
        else:
            size = final_buffer_start - start
        next_start = start + size
        schedule.append((start, next_start - 1))

    # Last stage: adaptation of fast parameters
    schedule.append((final_buffer_start, num_steps - 1))

    return schedule
