"""Kernel to compute the Gelman-Rubin convergence diagnostic (Rhat) online.
"""
from typing import NamedTuple

import jax.numpy as jnp

from mcx.inference.warmup.mass_matrix_adaptation import (
    WelfordAlgorithmState,
    welford_algorithm,
)


class GelmanRubinState(NamedTuple):
    w_state: WelfordAlgorithmState
    rhat: float


def online_gelman_rubin():
    """Online estimation of the Gelman-Rubin diagnostic."""

    w_init, w_update, w_covariance = welford_algorithm(True)

    def init(num_chains):
        w_state = w_init(num_chains)
        return GelmanRubinState(w_state, 0)

    def update(chain_state, rhat_state):
        within_state, step, num_chains, _, _, _ = rhat_state

        positions = chain_state.position
        within_state = w_update(within_state, positions)

        covariance, step, mean = w_covariance(rhat_state)
        within_var = jnp.mean(covariance)
        between_var = jnp.var(mean, ddof=1)
        estimator = ((step - 1) / step) * within_var + between_var
        rhat = jnp.sqrt(estimator / within_var)

        return GelmanRubinState(within_state, rhat)

    return init, update


def split_gelman_rubin():
    pass
