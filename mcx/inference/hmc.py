from typing import Any, Callable, NamedTuple, Tuple

import jax
from jax import numpy as np

from mcx.inference.integrators import velocity_verlet, hmc_proposal
from mcx.inference.kernels import HMCState, hmc_kernel
from mcx.inference.metrics import gaussian_euclidean_metric

# from mcx.inference.warmup import stan_hmc_warmup
from mcx.inference.warmups import stan_hmc_warmup, stan_warmup_schedule


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
    ) -> Tuple[HMCParameters, HMCState]:

        def generate_kernel(warmup_state):
            inverse_mass_matrix = warmup_state.mm_state.inverse_mass_matrix
            step_size = np.exp(warmup_state.da_state.log_step_size)
            momentum_generator, kinetic_energy = gaussian_euclidean_metric(
                inverse_mass_matrix
            )
            integrator_step = integrator(logpdf, kinetic_energy)
            proposal = hmc_proposal(integrator_step, step_size, num_integration_steps)
            kernel = hmc_kernel(proposal, momentum_generator, kinetic_energy, logpdf)
            return kernel

        schedule = stan_warmup_schedule(num_warmup_steps)

        init, update, final = stan_hmc_warmup(
            logpdf, integrator, generate_kernel, num_integration_steps, num_warmup_steps
        )

        initial_step_size = 0.1
        rng_keys, state, warmup_state = jax.vmap(init, in_axes=(None, 0, None))(
            rng_key, initial_state, initial_step_size
        )
        # NOTE: bugs because JAX seemingly does not like it when vmap
        # outputs a function. We may have to pass an HMCKernel object
        # around, or just the warmup state, actually
        # if you applied generate_kernel here you would still have to vmap
        # the function and it would not work.
        for step in range(num_warmup_steps):
            stage, _ = schedule[step]
            rng_keys, state, warmup_state = jax.vmap(update, in_axes=(0, None, 0, 0))(
                rng_keys, stage, state, warmup_state
            )

        print(state, warmup_state)

        """

        rng_keys, kernel, state, warmup_state = jax.vmap(init, in_axes=(None, 0, None))(
            rng_key, initial_state, initial_step_size
        )

        for _ in range(num_warmup_steps):
            rng_keys, kernel, state, warmup_state = jax.vmap(
                update, in_axes=(0, 0, 0, 0)
            )(rng_keys, kernel, state, warmup_state)

        print(warmup_state)

        kernel, state = jax.vmap(final, in_axes=(0, 0))(state, warmup_state)
        
        state, step_size, inverse_mass_matrix = jax.vmap(
            stan_hmc_warmup, in_axes=(None, None, 0, None, None, None, None, None, None)
        )(
            rng_key,
            logpdf,
            initial_state,
            gaussian_euclidean_metric,
            integrator,
            0.1,
            num_integration_steps,
            num_warmup_steps,
            is_mass_matrix_diagonal,
        )
        """

        parameters = HMCParameters(
            step_size, num_integration_steps, inverse_mass_matrix
        )

        return parameters, state

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
