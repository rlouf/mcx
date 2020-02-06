from functools import partial

import jax
import jax.numpy as np


@partial(jax.jit, static_argnums=(1, 2))
def rwm_kernel(rng_key, logpdf, move_scale, position, log_prob):
    """Random Walk Metropolis transition kernel.

    Moves the chains by one step using the Random Walk Metropolis algorithm.

    Args:
        rng_key: jax.random.PRNGKey
            Key for the pseudo random number generator.
        logpdf: function
            Returns the log-probability of the model given a position.
        move_scale: float
            Standard deviation of the Gaussian distribution from which the
            move proposals are sampled.
        position: jax.numpy.ndarray, shape (n_dims,)
            The starting position.
        log_prob: float
            The log probability at the starting position.

    Returns:
        The next position of the chains along with its log probability.
    """
    key_move, key_uniform = jax.random.split(rng_key)

    move_proposal = jax.random.normal(key_move, shape=position.shape) * move_scale
    proposal = position + move_proposal
    proposal_log_prob = logpdf(proposal)

    log_uniform = np.log(jax.random.uniform(key_uniform))
    do_accept = log_uniform < proposal_log_prob - log_prob

    position = np.where(do_accept, proposal, position)
    log_prob = np.where(do_accept, proposal_log_prob, log_prob)
    return position, log_prob
