"""Kernel to compute the Gelman-Rubin convergence diagnostic (Rhat) online.
"""
from typing import NamedTuple

import jax.numpy as np

from mcx.inference.adaptive import WelfordAlgorithmState, welford_algorithm


class GelmanRubinState(NamedTuple):
    within_welford: WelfordAlgorithmState
    rhat: float


def gelman_rubin():
    """Compute the Gelman-Rubin diagnostic."""

    w_init, w_update, w_covariance = welford_algorithm(True)

    def init(num_chains):
        w_state = w_init(num_chains)
        return GelmanRubinState(w_state, 0)

    def update(chain_state, rhat_state):
        within_state, step, num_chains, _, _, _ = rhat_state

        positions = chain_state.position
        within_state = w_update(within_state, positions)

        covariance, step, mean = w_covariance(rhat_state)
        within_var = np.mean(covariance)
        between_var = np.var(mean, ddof=1)
        estimator = ((step - 1) / step) * within_var + between_var
        rhat = np.sqrt(estimator / within_var)

        return GelmanRubinState(within_state, rhat)

    return init, update
