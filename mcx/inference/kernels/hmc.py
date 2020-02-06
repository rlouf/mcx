from functools import partial
from typing import Callable, Tuple, Union

import numpy
import jax
import jax.numpy as np
import jax.scipy.stats as st

Array = Union[numpy.ndarray, jax.numpy.DeviceArray]
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
    key_momentum, key_uniform = jax.random.split(rng_key)

    position, log_prob, log_prob_grad = state
    n_features = position.shape[0]  # may need to be updated as API matures

    momentum = jax.random.normal(key_momentum, (1, n_features))
    proposal, proposal_momentum, proposal_log_prob, proposal_log_prob_grad = integrator(
        position,
        momentum,
        log_prob_grad,
        log_prob,
        path_length=path_length,
        step_size=step_size,
    )

    # Metropolis Hastings acceptance step to correct for integration errors.
    initial_energy = np.sum(st.norm.logpdf(momentum)) - log_prob
    proposal_energy = np.sum(st.norm.logpdf(proposal_momentum)) - proposal_log_prob

    log_uniform = np.log(jax.random.uniform(key_uniform))
    do_accept = log_uniform < proposal_energy - initial_energy
    if do_accept:
        position = proposal
        log_prob = proposal_log_prob
        log_prob_grad = proposal_log_prob_grad

    state = (position, log_prob, log_prob_grad)

    return state
