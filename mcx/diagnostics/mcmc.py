"""Diagnostics that are specific to MCMC algorithms."""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp


class DivergencesState(NamedTuple):
    num_divergences: jnp.ndarray
    metric: int  # maximum number of divergences


def divergences() -> Tuple[str, Callable, Callable]:
    """Count the number of divergences.

    We keep a count of the current number of divergences for each chain and
    return the maximum number of divergences to be displayed.

    """
    metric_name = "divergences"

    def init(init_state) -> DivergencesState:
        """Initialize the divergence counters."""
        num_chains, _ = init_state.position.shape
        num_divergences = jnp.zeros(num_chains)
        return DivergencesState(num_divergences, 0)

    @jax.jit
    def update(_, info, divergence_state: DivergencesState) -> DivergencesState:
        """Update the number of divergences."""
        num_divergences, *_ = divergence_state
        is_divergent = info.is_divergent.astype(int)
        num_divergences = num_divergences + is_divergent
        max_num_divergences = jnp.max(num_divergences)
        return DivergencesState(num_divergences, max_num_divergences)

    return metric_name, init, update
