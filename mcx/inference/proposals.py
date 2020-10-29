"""Move proposals for Monte Carlo samplers."""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as np

from mcx.inference.integrators import Integrator, IntegratorState

# --------------------------------------------------------------------
#                       == Random Walk ==
# --------------------------------------------------------------------


def binary_proposal(p: np.DeviceArray) -> Callable:
    """Binary Random Walk proposal.

    Propose a new position that is one step away from the current positions.
    Suitable for discrete variables.

    """

    @jax.jit
    def propose(rng_key: jax.random.PRNGKey) -> int:
        return 2 * jax.random.bernoulli(rng_key, p) - 1

    return propose


def normal_proposal(sigma: np.DeviceArray) -> Callable:
    """Normal Random Walk proposal.

    Propose a new position such that its distance to the current position is
    normally distributed. Suitable for continuous variables.

    """

    @jax.jit
    def propose(rng_key: jax.random.PRNGKey, position: jax.numpy.DeviceArray) -> np.DeviceArray:
        step = jax.random.normal(rng_key, shape=np.shape(position)) * sigma
        new_position = position + step
        return new_position

    return propose


# --------------------------------------------------------------------
#                   == Hamiltonian Monte Carlo  ==
# --------------------------------------------------------------------


HMCProposalState = IntegratorState


class HMCProposalInfo(NamedTuple):
    step_size: float
    num_integration_steps: int


Proposer = Callable[
    [jax.random.PRNGKey, HMCProposalState], Tuple[HMCProposalState, HMCProposalInfo]
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


def empirical_hmc_proposal(
    integrator: Integrator, path_length_generator: Callable, step_size: float
) -> Proposer:
    """Proposal for the empirical HMC algorithm.

    The empirical HMC algorithm [1]_ uses an adaptive scheme for the path
    length: during warmup, a distribution of eligible path lengths is computed;
    the integrator draws a random path length value from this distribution each
    time it is called.

    References
    ----------
    .. [1]: Wu, Changye, Julien Stoehr, and Christian P. Robert. "Faster
            Hamiltonian Monte Carlo by learning leapfrog scale." arXiv preprint
            arXiv:1810.04449 (2018).

    """

    @jax.jit
    def propose(
        rng_key: jax.random.PRNGKey, state: HMCProposalState
    ) -> Tuple[HMCProposalState, HMCProposalInfo]:
        path_length = path_length_generator(rng_key)
        num_integration_steps = np.clip(path_length / step_size, a_min=1).astype(int)
        new_state = jax.lax.fori_loop(
            0,
            num_integration_steps,
            lambda i, state: integrator(state, step_size),
            state,
        )
        return new_state, HMCProposalInfo(step_size, num_integration_steps)

    return propose
