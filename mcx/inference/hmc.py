from typing import Any, Callable, NamedTuple, Tuple

import jax
from jax import numpy as np

from mcx.inference.integrators import velocity_verlet, hmc_proposal
from mcx.inference.kernels import HMCState, hmc_kernel
from mcx.inference.metrics import gaussian_euclidean_metric


class HMCParameters(NamedTuple):
    step_size: float
    num_integration_steps: int
    inverse_mass_matrix: np.DeviceArray


def HMC(
    step_size: float,
    num_integration_steps: int,
    inverse_mass_matrix: np.DeviceArray,
    integrator: Callable = velocity_verlet,
    is_mass_matrix_diagonal: bool = False,
) -> Tuple[Callable, Callable, Callable, Callable, Callable]:

    parameters = HMCParameters(step_size, num_integration_steps, inverse_mass_matrix)

    def init(position: np.DeviceArray, value_and_grad: Callable) -> HMCState:
        log_prob, log_prob_grad = value_and_grad(position)
        return HMCState(position, log_prob, log_prob_grad)

    def warmup(
        initial_state: HMCState, logpdf: Callable, num_warmup_steps: int
    ) -> Tuple[HMCParameters, HMCState]:
        return parameters, initial_state

    def build_kernel(logpdf: Callable, parameters: HMCParameters) -> Callable:
        """Builds the kernel that moves the chain from one point
        to the next.
        """

        potential = logpdf

        try:
            inverse_mass_matrix = parameters.inverse_mass_matrix
            num_integration_steps = parameters.num_integration_steps
            step_size = parameters.step_size
        except AttributeError:
            AttributeError(
                "The Hamiltonian Monte Carlo algorithm requires the following parameters: mass matrix, inverse mass matrix and step size."
            )

        momentum_generator, kinetic_energy = gaussian_euclidean_metric(
            inverse_mass_matrix,
        )
        integrator_step = integrator(potential, kinetic_energy)
        proposal = hmc_proposal(integrator_step, step_size, num_integration_steps)
        kernel = hmc_kernel(proposal, momentum_generator, kinetic_energy, potential)

        return kernel

    def adapt_loglikelihood(logpdf: Callable) -> Callable:
        """Potential is minus the loglikelihood."""

        def potential(array: np.DeviceArray) -> float:
            return -logpdf(array)

        return potential

    def to_trace(chain: Any, ravel_fn: Callable) -> dict:
        """Translate the raw chains to a format that can be understood by and
        is useful to humans.
        """

        trace = {}

        def ravel_chain(chain):
            return jax.vmap(ravel_fn, in_axes=(0,))(chain)

        positions_array = np.stack([state.position for state, _ in chain], axis=1)
        trace["posterior"] = jax.vmap(ravel_chain, in_axes=(0,))(positions_array)

        trace["log_likelihood"] = np.stack(
            [state.log_prob for state, _ in chain], axis=1
        )

        trace["info"] = {}
        trace["info"]["is_divergent"] = np.stack(
            [info.is_divergent for _, info in chain], axis=1
        )
        trace["info"]["is_accepted"] = np.stack(
            [info.is_accepted for _, info in chain], axis=1
        )

        return trace

    return init, warmup, build_kernel, to_trace, adapt_loglikelihood
