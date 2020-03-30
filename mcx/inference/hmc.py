from typing import NamedTuple

from jax import numpy as np

from mcx.inference.integrators import velocity_verlet, hmc_proposal
from mcx.inference.kernels import HMCState, hmc_kernel
from mcx.inference.metrics import gaussian_euclidean_metric


class HMCParameters(NamedTuple):
    step_size: float
    num_integration_steps: float
    mass_matrix_sqrt: np.DeviceArray
    inverse_mass_matrix: np.DeviceArray


def HMC(
    step_size=None,
    num_integration_steps=None,
    mass_matrix_sqrt=None,
    inverse_mass_matrix=None,
    integrator=velocity_verlet,
    is_mass_matrix_diagonal=False,
):

    parameters = HMCParameters(
        step_size, num_integration_steps, mass_matrix_sqrt, inverse_mass_matrix
    )

    def init(position, value_and_grad):
        log_prob, log_prob_grad = value_and_grad(position)
        return HMCState(position, log_prob, log_prob_grad)

    def warmup(initial_state, logpdf, num_warmup_steps):
        return parameters, initial_state

    def build_kernel(logpdf, parameters):
        """Builds the kernel that moves the chain from one point
        to the next.
        """

        try:
            mass_matrix_sqrt = parameters.mass_matrix_sqrt
            inverse_mass_matrix = parameters.inverse_mass_matrix
            num_integration_steps = parameters.num_integration_steps
            step_size = parameters.step_size
        except AttributeError:
            AttributeError(
                "The Hamiltonian Monte Carlo algorithm requires the following parameters: mass matrix, inverse mass matrix and step size."
            )

        momentum_generator, kinetic_energy = gaussian_euclidean_metric(
            mass_matrix_sqrt, inverse_mass_matrix,
        )
        integrator_step = integrator(logpdf, kinetic_energy)
        proposal = hmc_proposal(integrator_step, step_size, num_integration_steps)
        kernel = hmc_kernel(proposal, momentum_generator, kinetic_energy, logpdf)

        return kernel

    def to_trace(states_chain, ravel_fn):
        """ Translate the raw chains to a format that can be understood by and
        is useful to humans.
        """
        return states_chain

    return init, warmup, build_kernel, to_trace
