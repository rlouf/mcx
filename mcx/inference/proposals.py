"""Move proposals for Monte Carlo samplers."""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from mcx.inference.integrators import Integrator, IntegratorState

# --------------------------------------------------------------------
#                   == Hamiltonian Monte Carlo  ==
# --------------------------------------------------------------------


HMCProposalState = IntegratorState


class HMCProposalInfo(NamedTuple):
    step_size: float
    num_integration_steps: int


Proposer = Callable[
    [jnp.ndarray, HMCProposalState], Tuple[HMCProposalState, HMCProposalInfo]
]


def hmc_proposal(
    integrator: Integrator, step_size: float, num_integration_steps: int = 1
) -> Proposer:
    """Vanilla HMC proposal.

    Given a path length and a step size, the HMC proposer will run the
    appropriate number of integration steps (typically with the velocity Verlet
    algorithm).

    """
    info = HMCProposalInfo(step_size, num_integration_steps)

    @jax.jit
    def propose(
        _, init_state: HMCProposalState
    ) -> Tuple[HMCProposalState, HMCProposalInfo]:
        new_state = jax.lax.fori_loop(
            0,
            num_integration_steps,
            lambda i, state: integrator(state, step_size),
            init_state,
        )
        return new_state, info

    return propose
