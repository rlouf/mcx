from datetime import datetime
from typing import Callable, Dict, NamedTuple, Tuple, Optional
import warnings

import jax
from jax import numpy as np
from tqdm import tqdm

from mcx.inference.integrators import velocity_verlet, hmc_proposal
from mcx.inference.kernels import HMCInfo, HMCState, hmc_kernel, hmc_init
from mcx.inference.metrics import gaussian_euclidean_metric
from mcx.inference.warmup import stan_hmc_warmup, stan_warmup_schedule


class HMCParameters(NamedTuple):
    num_integration_steps: int
    step_size: Optional[float]
    inverse_mass_matrix: Optional[np.DeviceArray]


class HMC:
    def __init__(
        self,
        num_integration_steps: int = 10,
        step_size: float = None,
        inverse_mass_matrix: np.DeviceArray = None,
        is_mass_matrix_diagonal: bool = False,
        integrator: Callable = velocity_verlet,
    ):
        self.needs_warmup = True
        if step_size is not None and inverse_mass_matrix is None:
            warnings.warn(
                "You specified the step size for the HMC algorithm "
                "but not the mass matrix. MCX is currently unable to "
                "run the warmup on a single variable and thus will run "
                "the Stan warmup for both step size and mass matrix. If this is not "
                "what you want, please instantiate the kernel with both a step size "
                "and a mass matrix.",
                UserWarning,
            )
        elif step_size is None and inverse_mass_matrix is not None:
            warnings.warn(
                "You specified the mass matrix for the HMC algorithm "
                "but not the step size. MCX is currently unable to "
                "run the warmup on a single variable and thus will run "
                "the Stan warmup for both step size and mass matrix. If this is not "
                "what you want, please instantiate the kernel specifying both the step size "
                "and the mass matrix.",
                UserWarning,
            )
        elif step_size is not None and inverse_mass_matrix is not None:
            self.needs_warmup = False

        self.integrator = integrator
        self.is_mass_matrix_diagonal = is_mass_matrix_diagonal
        self.parameters = HMCParameters(
            num_integration_steps, step_size, inverse_mass_matrix
        )

    def states(self, positions, loglikelihood):
        potential = self._to_potential(loglikelihood)
        potential_val_and_grad = jax.value_and_grad(potential)
        states = jax.vmap(hmc_init, in_axes=(0, None))(
            positions, potential_val_and_grad
        )
        return states

    def warmup(
        self,
        rng_key: jax.random.PRNGKey,
        initial_state: HMCState,
        kernel_factory: Callable,
        num_chains,
        num_warmup_steps: int = 1000,
        accelerate=False,
        initial_step_size: float = 0.1,
    ) -> Tuple[HMCState, HMCParameters]:
        """I don't like having a ton of warmup logic in here."""

        if not self.needs_warmup:
            parameters = HMCParameters(
                np.ones(initial_state.position.shape[0], dtype=np.int32)
                * self.parameters.num_integration_steps,
                np.ones(initial_state.position.shape[0]) * self.parameters.step_size,
                np.array(
                    [
                        self.parameters.inverse_mass_matrix
                        for _ in range(initial_state.position.shape[0])
                    ]
                ),
            )
            return initial_state, parameters

        # kernel_factory = self.kernel_factory(loglikelihood)
        hmc_factory = jax.partial(kernel_factory, self.parameters.num_integration_steps)
        init, update, final = stan_hmc_warmup(hmc_factory, self.is_mass_matrix_diagonal)

        rng_keys = jax.random.split(rng_key, num_chains)
        chain_state, warmup_state = jax.vmap(init, in_axes=(0, 0, None))(
            rng_keys, initial_state, initial_step_size
        )

        schedule = np.array(stan_warmup_schedule(num_warmup_steps))

        if accelerate:

            print(
                f"sampler: warmup {num_chains:,} chains for {num_warmup_steps:,} iterations.",
                end=" ",
            )
            start = datetime.now()

            @jax.jit
            def update_chain(carry, interval):
                rng_key, chain_state, warmup_state = carry
                stage, is_middle_window_end = interval

                _, rng_key = jax.random.split(rng_key)
                keys = jax.random.split(rng_key, num_chains)
                chain_state, warmup_state = jax.vmap(
                    update, in_axes=(0, None, None, 0, 0)
                )(keys, stage, is_middle_window_end, chain_state, warmup_state,)

                return (rng_key, chain_state, warmup_state), (chain_state, warmup_state)

            last_state, _ = jax.lax.scan(
                update_chain, (rng_key, chain_state, warmup_state), schedule
            )
            _, chain_state, warmup_state = last_state

            print(f"Done in {(datetime.now()-start).total_seconds():.1f} seconds.")

        else:

            chain = []
            with tqdm(schedule, unit="samples") as progress:
                progress.set_description(
                    f"Warming up {num_chains} chains for {num_warmup_steps} steps",
                    refresh=False,
                )
                for interval in progress:
                    _, rng_key = jax.random.split(rng_key)
                    rng_keys = jax.random.split(rng_key, num_chains)
                    stage, is_middle_window_end = interval
                    chain_state, warmup_state = jax.vmap(
                        update, in_axes=(0, None, None, 0, 0)
                    )(rng_keys, stage, is_middle_window_end, chain_state, warmup_state)
                    chain.append((chain_state, warmup_state))

            chain_state, warmup_state = chain[
                -1
            ]  # not using it now, but to give lax.scan a fair comparison

        step_size, inverse_mass_matrix = jax.vmap(final, in_axes=(0,))(warmup_state)
        num_integration_steps = self.parameters.num_integration_steps

        parameters = HMCParameters(
            np.ones(initial_state.position.shape[0], dtype=np.int32)
            * num_integration_steps,
            step_size,
            inverse_mass_matrix,
        )

        return chain_state, parameters

    def kernel_factory(self, loglikelihood: Callable) -> Callable:
        potential = self._to_potential(loglikelihood)

        def build_kernel(num_integration_steps, step_size, inverse_mass_matrix):
            momentum_generator, kinetic_energy = gaussian_euclidean_metric(
                inverse_mass_matrix,
            )
            integrator_step = self.integrator(potential, kinetic_energy)
            proposal = hmc_proposal(integrator_step, step_size, num_integration_steps)
            kernel = hmc_kernel(proposal, momentum_generator, kinetic_energy, potential)
            return kernel

        return build_kernel

    def make_trace(self, chain: Tuple[HMCState, HMCInfo], ravel_fn: Callable,) -> Dict:
        """Translate the raw chain into a Trace object.

        Parameters
        ----------
        chain
            A tuple that contains the HMC sampler's states and additional information
            on the sampling process.

        Returns
        -------
        A trace that contains the model's posterior samples and relevant
        sampling information.

        """
        state, info = chain

        # We unravelled positions before sampling to be able to make
        # computations on flat arrays in the backend. We now need to bring
        # their to their original shape before adding to the trace.
        ravel_chain = jax.vmap(ravel_fn, in_axes=(0,))
        samples = jax.vmap(ravel_chain, in_axes=(1,))(state.position)

        sampling_info = {
            "potential_energy": state.potential_energy.T,
            "acceptance_probability": info.acceptance_probability.T,
            "is_divergent": info.is_divergent.T,
            "energy": info.energy.T,
            "step_size": info.proposal_info.step_size.T,
            "num_integration_steps": info.proposal_info.num_integration_steps.T,
        }

        return samples, sampling_info

    def _to_potential(self, loglikelihood: Callable) -> Callable:
        """The potential in the Hamiltonian Monte Carlo algorithm is equal to
        minus the log-likelihood.

        """

        def potential(array: np.DeviceArray) -> float:
            return -loglikelihood(array)

        return potential
