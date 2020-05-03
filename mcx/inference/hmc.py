from typing import NamedTuple

import jax
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

        potential = logpdf

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
        integrator_step = integrator(potential, kinetic_energy)
        proposal = hmc_proposal(integrator_step, step_size, num_integration_steps)
        kernel = hmc_kernel(proposal, momentum_generator, kinetic_energy, potential)

        return kernel

    def adapt_loglikelihood(logpdf):
        """Potential is minus the loglikelihood.
        """
        def potential(array):
            return - logpdf(array)
        return potential

    def to_trace(chain, ravel_fn):
        """ Translate the raw chains to a format that can be understood by and
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
