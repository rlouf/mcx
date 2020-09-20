"""Implementation of the Stan warmup for the HMC family of sampling algorithms."""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as np

from mcx.inference.warmup.step_size_adaptation import (
    DualAveragingState,
    dual_averaging,
    find_reasonable_step_size,
)
from mcx.inference.warmup.mass_matrix_adaptation import (
    MassMatrixAdaptationState,
    mass_matrix_adaptation,
)


__all__ = ["stan_hmc_warmup", "stan_warmup_schedule"]


class StanWarmupState(NamedTuple):
    da_state: DualAveragingState
    mm_state: MassMatrixAdaptationState


def stan_hmc_warmup(kernel_generator, num_warmup_steps: int):
    """Warmup scheme for sampling procedures based on euclidean manifold HMC.
    The schedule and algorithms used match Stan's [1]_ as closely as possible.

    Unlike several other libraries, we separate the warmup and sampling phases
    explicitly. This ensure a better modularity; a change in the warmup does
    not affect the sampling. It also allows users to run their own warmup
    should they want to.

    Stan's warmup consists in the three following phases:

    1. A fast adaptation window where only the step size is adapted using
    Nesterov's dual averaging scheme to match a target acceptance rate.
    2. A succession of slow adapation windows (where the size of a window
    is double that of the previous window) where both the mass matrix and the step size
    are adapted. The mass matrix is recomputed at the end of each window; the step
    size is re-initialized to a "reasonable" value.
    3. A last fast adaptation window where only the step size is adapted.

    """
    first_stage_init, first_stage_update = stan_first_stage(kernel_generator)
    second_stage_init, second_stage_update, second_stage_final = stan_second_stage(
        kernel_generator
    )

    def init(rng_key, initial_state, initial_step_size):
        mm_state = second_stage_init(initial_state)

        da_state = first_stage_init(
            rng_key, mm_state.inverse_mass_matrix, initial_state, initial_step_size,
        )

        warmup_state = StanWarmupState(da_state, mm_state)

        return rng_key, initial_state, warmup_state

    @jax.jit
    def update(rng_key, stage, is_middle_window_end, state, warmup_state):
        """Only update the step size at first."""
        _, rng_key = jax.random.split(rng_key)

        step_size = np.exp(warmup_state.da_state.log_step_size)
        inverse_mass_matrix = warmup_state.mm_state.inverse_mass_matrix
        kernel = kernel_generator(step_size, inverse_mass_matrix)

        state, info = kernel(rng_key, state)

        rng_key, state, warmup_state = jax.lax.switch(
            stage,
            (first_stage_update, second_stage_update),
            (rng_key, state, warmup_state),
        )

        rng_key, state, warmup_state = jax.lax.cond(
            is_middle_window_end,
            second_stage_final,
            lambda x: x,
            (rng_key, state, warmup_state),
        )

        return rng_key, state, warmup_state

    def final(warmup_state):
        step_size = np.exp(warmup_state.da_state.log_step_size)
        inverse_mass_matrix = warmup_state.mm_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final


@partial(jax.jit, static_argnums=(0,))
def stan_first_stage(kernel_generator):
    """First stage of the Stan warmup. Only the step size is adapted using
    Nesterov's dual averaging algorithms.
    """

    da_init, da_update = dual_averaging()

    def init(
        rng_key, inverse_mass_matrix, initial_state, initial_step_size,
    ):
        kernel_from_step_size = jax.partial(
            kernel_generator, inverse_mass_matrix=inverse_mass_matrix
        )
        step_size = find_reasonable_step_size(
            rng_key, kernel_from_step_size, initial_state, initial_step_size,
        )
        da_state = da_init(step_size)
        return da_state

    @jax.jit
    def update(state):
        rng_key, chain_state, warmup_state = state
        _, rng_key = jax.random.split(rng_key)

        step_size = np.exp(warmup_state.da_state.log_step_size)
        inverse_mass_matrix = warmup_state.mm_state.inverse_mass_matrix
        kernel = kernel_generator(step_size, inverse_mass_matrix)

        chain_state, info = kernel(rng_key, chain_state)

        new_da_state = da_update(info.acceptance_probability, warmup_state.da_state)
        new_warmup_state = StanWarmupState(new_da_state, warmup_state.mm_state)

        return rng_key, chain_state, new_warmup_state

    return init, update


@partial(jax.jit, static_argnums=(0,))
def stan_second_stage(kernel_generator, is_mass_matrix_diagonal: bool = True):
    """Slow stage of the Stan warmup.

    In this stage we adapt the mass matrix as the inverse of the chain's
    covariance. The step size and mass matrix are re-initialized at the end of
    each window.
    """

    mm_init, mm_update, mm_final = mass_matrix_adaptation(is_mass_matrix_diagonal)
    da_init, _ = dual_averaging()

    def init(chain_state):
        n_dims = np.shape(chain_state.position)[-1]
        mm_state = mm_init(n_dims)
        return mm_state

    @jax.jit
    def update(state):
        """The kernel does not change, we are just updating the chain's
        covariance.
        """
        rng_key, chain_state, warmup_state = state
        _, rng_key = jax.random.split(rng_key)

        step_size = np.exp(warmup_state.da_state.log_step_size)
        inverse_mass_matrix = warmup_state.mm_state.inverse_mass_matrix
        kernel = kernel_generator(step_size, inverse_mass_matrix)

        chain_state, _ = kernel(rng_key, chain_state)
        new_mm_state = mm_update(warmup_state.mm_state, chain_state.position)
        new_warmup_state = StanWarmupState(warmup_state.da_state, new_mm_state)
        return rng_key, chain_state, new_warmup_state

    @jax.jit
    def final(state):
        rng_key, chain_state, warmup_state = state

        new_mm_state = mm_final(warmup_state.mm_state)
        step_size_to_kernel = jax.partial(
            kernel_generator, inverse_mass_matrix=new_mm_state.inverse_mass_matrix
        )

        step_size = np.exp(warmup_state.da_state.log_step_size)
        step_size = find_reasonable_step_size(
            rng_key, step_size_to_kernel, chain_state, step_size,
        )

        da_state = da_init(step_size)

        new_warmup_state = StanWarmupState(da_state, new_mm_state)

        return rng_key, chain_state, new_warmup_state

    return init, update, final


def stan_warmup_schedule(
    num_steps, initial_buffer_size=75, first_window_size=25, final_buffer_size=50
):
    """Returns an adaptation warmup schedule.

    The schedule below is intended to be as close as possible to Stan's _[1].
    The warmup period is split into three stages:

    1. An initial fast interval to reach the typical set.
    2. "Slow" parameters that require global information (typically covariance)
       are estimated in a series of expanding windows with no memory.
    3. Fast parameters are learned after the adaptation of the slow ones.

    See _[1] for a more detailed explanation.

    The "fast" stage is given the index 0 while the slow "stage" is given the
    index 1.

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
        schedule += [(0, False)] * (num_steps - 1)
    else:
        if initial_buffer_size + first_window_size + final_buffer_size > num_steps:
            initial_buffer_size = int(0.15 * num_steps)
            final_buffer_size = int(0.1 * num_steps)
            first_window_size = num_steps - initial_buffer_size - final_buffer_size

        # First stage: adaptation of fast parameters
        schedule += [(0, False)] * (initial_buffer_size - 1)
        schedule.append((0, False))

        # Second stage: adaptation of slow parameters
        final_buffer_start = num_steps - final_buffer_size

        next_window_size = first_window_size
        next_window_start = initial_buffer_size
        while next_window_start < final_buffer_start:
            current_start, current_size = next_window_start, next_window_size
            if 3 * current_size <= final_buffer_start - current_start:
                next_window_size = 2 * current_size
            else:
                current_size = final_buffer_start - current_start
            next_window_start = current_start + current_size
            schedule += [(1, False)] * (next_window_start - 1 - current_start)
            schedule.append((1, True))

        # Last stage: adaptation of fast parameters
        schedule += [(0, False)] * (num_steps - 1 - final_buffer_start)
        schedule.append((0, False))

    return schedule
