# encoding: utf-8
"""
Sampling methods using the Random Walk Metropolis algorithm.
"""
from functools import partial
from typing import Callable, Generator

import jax

from mcx.inference.kernels import HMCState

KernelState = HMCState


@partial(jax.jit, static_argnums=(1, 2, 3))
def single_chain_stepper(
    rng_key: jax.random.PRNGKey,
    kernel: Callable,
    initial_state: KernelState,
    n_steps: int,
) -> KernelState:
    """Applies a kernel several times along a single trajectory
    and returns the last state of the chain.
    """

    def update(_, rng_key, state):
        _, rng_key = jax.random.split(rng_key)
        new_state = kernel(rng_key, state)
        return rng_key, new_state

    state = jax.lax.fori_loop(0, n_steps, update, (rng_key, initial_state))
    return state


@partial(jax.jit, static_argnums=(1, 2, 3))
def single_chain_accumulator(
    rng_key: jax.random.PRNGKey,
    kernel: Callable,
    initial_state: KernelState,
    n_samples: int,
) -> jax.numpy.DeviceArray:
    """Applies the kernel several times along a single trajectory
    and returns all the intermediate states.
    """

    def update(state, rng_key):
        new_state = kernel(rng_key, state)
        return new_state, state

    keys = jax.random.split(rng_key, n_samples)
    states = jax.lax.scan(update, initial_state, keys)
    return states


def batch_sampler(
    rng_key: jax.random.PRNGKey,
    kernel: Callable,
    initial_states: jax.numpy.DeviceArray,
    n_samples: int,
) -> jax.numpy.DeviceArray:
    n_chains = initial_states.shape[0]
    rng_keys = jax.random.split(rng_key, n_chains)
    states = jax.vmap(single_chain_accumulator, in_axes=(0, None, 0, None), out_axes=0)(
        rng_keys, kernel, initial_states, n_samples
    )
    return states


def batch_stepper(
    rng_key: jax.random.PRNGKey,
    kernel: Callable,
    initial_states: jax.numpy.DeviceArray,
    n_steps: int,
) -> jax.numpy.DeviceArray:
    n_chains = initial_states.shape[0]
    rng_keys = jax.random.split(rng_key, n_chains)
    states = jax.vmap(single_chain_stepper, in_axes=(0, None, 0, None), out_axes=0)(
        rng_keys, kernel, initial_states, n_steps
    )
    return states


def batch_generator(
    rng_key: jax.random.PRNGKey,
    kernel: Callable,
    initial_states: jax.random.DeviceArray,
) -> Generator[KernelState, None, None]:
    n_chains = initial_states.shape[0]
    states = initial_states
    yield initial_states

    while True:
        rng_key, sample_key = jax.random.split(rng_key)
        chains_keys = jax.random.split(sample_key, n_chains)
        states = jax.vmap(kernel, in_axes=(0, 0), out_axes=(0,))(chains_keys, states)
        yield states
