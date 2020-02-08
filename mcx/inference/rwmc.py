# encoding: utf-8
"""
Sampling methods using the Random Walk Metropolis algorithm.
"""
from functools import partial
from typing import Callable, Generator

import jax
import numpy

from mcx.inference.kernels import rwm_kernel


def rw_metropolis_generator(
    rng_key: jax.random.PRNGKey,
    logpdf: Callable,
    initial_position: jax.numpy.DeviceArray,
    move_scale: float = 0.1,
) -> Generator[numpy.ndarray, None, None]:
    """Returns one sample at a time from an arbitrary number of chains using
    the Random Walk Metropolis algorithm.

    Notes:
        Using `rw_metropolis_generator` instead of `rw_metropolis_sampler` adds
        a computational overhead: it will take roughly twice as much time. If
        you are not in an exploration phase and you do not have a dynamic
        stopping rule you are probably better off using `rw_metropolis_sampler`
        with a fixed number of steps.

    Args:
        logpdf: The target log-probability density function to sample from.
        initial_position: shape (n_chain, n_features)
            The initial position of the random walk. The number of chains that
            will be computed is controlled by the shape of this tensor.
        n_samples: The number of samples to draw using the random walk metropolis
            algorithm.

    Yields:
        The next sample for each chain. Shape (n_chain, n_features).
    """
    n_chains = initial_position.shape[0]
    position = initial_position
    log_prob = jax.vmap(logpdf, in_axes=(1,))(initial_position)
    yield numpy.as_array(position)

    while True:
        rng_key, sample_key = jax.random.split(rng_key)
        chains_keys = jax.random.split(sample_key, n_chains)
        position, log_prob = jax.vmap(rwm_kernel, in_axes=(0, None, 0, 0), out_axes=0)(
            chains_keys, logpdf, move_scale, position, log_prob
        )
        yield numpy.as_array(position)


def rw_metropolis_sampler(
    rng_key: jax.random.PRNGKey,
    logpdf: Callable,
    initial_position: jax.numpy.ndarray,
    n_samples: int = 1000,
    move_scale: float = 0.1,
) -> numpy.ndarray:
    """Generates an arbitrary number of samples for an arbitrary number of
    chains using the Random Walk Metropolis algorithm.

    Args:
        logpdf: The target log-probability density function to sample from.
        initial_position: shape (n_chain, n_features)
            The initial position of the random walk. The number of chains that
            will be computed is controlled by the shape of this tensor.
        n_samples: The number of samples to draw using the random walk metropolis algorithm.

    Returns:
        The samples for each chain. Shape (n_chains, n_features, n_samples). #(TODO): check the shape
    """
    n_chains = initial_position.shape[0]
    rng_keys = jax.random.split(rng_key, n_chains)  # (nchains,)
    run_mcmc = jax.vmap(
        rw_metropolis_single_chain, in_axes=(0, None, None, 0), out_axes=0
    )
    position = run_mcmc(rng_keys, n_samples, logpdf, initial_position)
    return numpy.as_array(position)


@partial(jax.jit, static_argnums=(1, 2, 3))
def rw_metropolis_single_chain(
    rng_key, n_samples, move_scale, logpdf, initial_position
):
    """Generate samples for a single chain using the Random Walk Metropolis
    algorithm. Used for the sampler.

    Args:
        rng_key: jax.random.PRNGKey
            Key for the pseudo random number generator.
        n_samples: int
            Number of samples to generate per chain.
        logpdf: function
          Returns the log-probability of the model given a position.
        move_scale: float
            Standard deviation of the Gaussian distribution from which the
            move proposals are sampled.
        inital_position: jax.numpy.ndarray (n_dims, n_chains)
          The starting position.

    Returns:
        A single chain of samples (n_samples, n_dim)
    """

    def mh_update(_, state):
        key, position, log_prob = state
        _, key = jax.random.split(key)
        position, log_prob = rwm_kernel(key, logpdf, move_scale, position, log_prob)
        return (key, position, log_prob)

    position = initial_position
    log_prob = logpdf(initial_position)
    _, position, _ = jax.lax.fori_loop(
        0, n_samples, mh_update, (rng_key, position, log_prob)
    )
    return position
