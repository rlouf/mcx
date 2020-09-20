from functools import partial
from typing import Any, Callable, NamedTuple, Tuple

import jax
from jax import numpy as np
from tqdm import trange

from mcx.inference.integrators import velocity_verlet, hmc_proposal
from mcx.inference.kernels import HMCState, hmc_kernel
from mcx.inference.metrics import gaussian_euclidean_metric

from mcx.inference.warmup import stan_hmc_warmup, stan_warmup_schedule


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
    def init(position: np.DeviceArray, value_and_grad: Callable) -> HMCState:
        log_prob, log_prob_grad = value_and_grad(position)
        return HMCState(position, log_prob, log_prob_grad)

    def warmup(
        rng_key: jax.random.PRNGKey,
        initial_state: HMCState,
        logpdf: Callable,
        num_warmup_steps: int,
        initial_step_size: float = 0.1,
    ) -> Tuple[HMCParameters, HMCState]:
        """It would be nice to return the warmup trace here.
        """
        def generate_kernel(step_size: float, inverse_mass_matrix: np.DeviceArray):
            """May be obtained by partial application of `build_kernel`.
            There is duplication here. Problem is we don't need to raise an
            error when building the kernel here.
            """
            momentum_generator, kinetic_energy = gaussian_euclidean_metric(
                inverse_mass_matrix
            )
            integrator_step = integrator(logpdf, kinetic_energy)
            proposal = hmc_proposal(integrator_step, step_size, num_integration_steps)
            kernel = hmc_kernel(proposal, momentum_generator, kinetic_energy, logpdf)
            return kernel

        schedule = stan_warmup_schedule(num_warmup_steps)
        init, update, final = stan_hmc_warmup(generate_kernel, num_warmup_steps)

        rng_keys, state, warmup_state = jax.vmap(init, in_axes=(None, 0, None))(
            rng_key, initial_state, initial_step_size
        )

        for step in trange(num_warmup_steps):
            stage, is_middle_window_end = schedule[step]
            rng_keys, state, warmup_state = jax.vmap(
                update, in_axes=(0, None, None, 0, 0)
            )(rng_keys, stage, is_middle_window_end, state, warmup_state)

        step_size, inverse_mass_matrix = jax.vmap(final, in_axes=(0,))(warmup_state)

        parameters = HMCParameters(
            step_size, np.ones(initial_state.position.shape[0], dtype=np.int32) * num_integration_steps, inverse_mass_matrix,
        )

        return parameters, state

    def build_kernel(logpdf: Callable):#, parameters: HMCParameters) -> Callable:
        """Builds the kernel that moves the chain from one point
        to the next.
        """

        potential = logpdf

        def init_kernel(parameters):
            inverse_mass_matrix = parameters.inverse_mass_matrix
            num_integration_steps = parameters.num_integration_steps
            step_size = parameters.step_size
            momentum_generator, kinetic_energy = gaussian_euclidean_metric(
                inverse_mass_matrix,
            )
            integrator_step = integrator(potential, kinetic_energy)
            proposal = hmc_proposal(integrator_step, step_size, num_integration_steps)
            kernel = hmc_kernel(proposal, momentum_generator, kinetic_energy, potential)
            return kernel
        
        return init_kernel

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
