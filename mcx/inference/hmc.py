# encoding: utf-8
"""
Sampling methods using the Random Walk Metropolis algorithm.
"""
from functools import partial
from typing import Callable, Generator, Tuple, Union

import jax
import numpy

from mcx.inference.kernels import hmc_kernel

Array = Union[numpy.ndarray, jax.numpy.DeviceArray]


HMCState = Tuple[Array, Array, Array]


def hmc(
    rng_key: jax.random.RNGKey,
    logpdf: Callable,
    integrator: Callable,
    momentum_generator: Callable,
    kinetic_energy: Callable,
    position: jax.numpy.DeviceArray,
    step_size: float,
    path_length: float,
):
    """Returns a HMC kernel that moves a chain by one step.
    """

    def step(rng_key, state: HMCState) -> HMCState:
        position, log_prob, log_prob_grad = state
        _, rng_key = jax.random.split(rng_key)

        position, log_prob, log_prob_grad = hmc_kernel(
            rng_key, logpdf, integrator, path_length, step_size, state
        )
        return position, log_prob, log_prob_grad

    return step


@partial(jax.jit, static_argnums=(1, 2, 3))
def single_chain_stepper(
    rng_key, kernel, initial_state, n_steps
) -> jax.numpy.DeviceArray:
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
def single_chain_sampler(
    rng_key, kernel, initial_state, n_samples
) -> jax.numpy.DeviceArray:
    """Applies the kernel several times along a single trajectory
    and returns all the intermediate states.
    """

    def update(state, rng_key):
        new_state = kernel(rng_key, state)
        return new_state, state

    keys = jax.random.split(rng_key, n_samples)
    state = jax.lax.scan(update, initial_state, keys)
    return state


def hmc_generator(
    rng_key: jax.random.PRNGKey,
    logpdf: Callable,
    integrator: Callable,
    initial_position: jax.numpy.DeviceArray,
    initial_step_size: float = 0.1,
    path_length: int = 1,
) -> Generator[numpy.ndarray, None, None]:
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
    yield numpy.as_array(position)

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
        yield numpy.as_array(position)


def hmc_sampler(
    rng_key: jax.random.PRNGKey,
    logpdf: Callable,
    integrator: Callable,
    initial_position: jax.numpy.ndarray,
    n_samples: int = 1000,
    initial_step_size: float = 0.1,
    path_length: int = 1,
) -> numpy.ndarray:

    n_chains = initial_position.shape[0]
    rng_keys = jax.random.split(rng_key, n_chains)
    position = jax.vmap(hmc_single_chain, in_axes=(0, None, None, None, 0), out_axes=0)(
        rng_keys, n_samples, logpdf, integrator, initial_position
    )
    return numpy.as_array(position)


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
