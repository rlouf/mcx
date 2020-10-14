import warnings
from datetime import datetime
from typing import Callable, Dict, NamedTuple, Optional, Tuple

import jax
from jax import numpy as np
from tqdm import tqdm

from mcx.inference.integrators import hmc_proposal, velocity_verlet
from mcx.inference.kernels import HMCInfo, HMCState, hmc_init, hmc_kernel
from mcx.inference.metrics import gaussian_euclidean_metric
from mcx.inference.warmup import StanWarmupState, stan_hmc_warmup, stan_warmup_schedule


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
    ) -> Tuple[HMCState, HMCParameters, Optional[StanWarmupState]]:
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
            return initial_state, parameters, None

        hmc_factory = jax.partial(kernel_factory, self.parameters.num_integration_steps)
        init, update, final = stan_hmc_warmup(hmc_factory, self.is_mass_matrix_diagonal)

        rng_keys = jax.random.split(rng_key, num_chains)
        chain_state = initial_state
        warmup_state = jax.vmap(init, in_axes=(0, 0, None))(
            rng_keys, chain_state, initial_step_size
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
                chain_state, warmup_state, chain_info = jax.vmap(
                    update, in_axes=(0, None, None, 0, 0)
                )(keys, stage, is_middle_window_end, chain_state, warmup_state)

                return (
                    (rng_key, chain_state, warmup_state),
                    (chain_state, warmup_state, chain_info),
                )

            last_state, warmup_chain = jax.lax.scan(
                update_chain, (rng_key, chain_state, warmup_state), schedule
            )
            _, last_chain_state, last_warmup_state = last_state

            print(f"Done in {(datetime.now()-start).total_seconds():.1f} seconds.")

        else:

            @jax.jit
            def update_fn(rng_key, interval, chain_state, warmup_state):
                rng_keys = jax.random.split(rng_key, num_chains)
                stage, is_middle_window_end = interval
                chain_state, warmup_state, chain_info = jax.vmap(
                    update, in_axes=(0, None, None, 0, 0)
                )(rng_keys, stage, is_middle_window_end, chain_state, warmup_state)
                return chain_state, warmup_state, chain_info

            chain = []
            with tqdm(schedule, unit="samples") as progress:
                progress.set_description(
                    f"Warming up {num_chains} chains for {num_warmup_steps} steps",
                    refresh=False,
                )
                for interval in progress:
                    _, rng_key = jax.random.split(rng_key)
                    chain_state, warmup_state, chain_info = update_fn(
                        rng_key, interval, chain_state, warmup_state
                    )
                    chain.append((chain_state, warmup_state, chain_info))

            last_chain_state, last_warmup_state, _ = chain[-1]

            # The sampling process, the composition between scan and for loop
            # is identical for the warmup and the sampling.  Should we
            # generalize this to only call a single `scan` function?
            stack = lambda y, *ys: np.stack((y, *ys))
            warmup_chain = jax.tree_multimap(stack, *chain)

        step_size, inverse_mass_matrix = jax.vmap(final, in_axes=(0,))(
            last_warmup_state
        )
        num_integration_steps = self.parameters.num_integration_steps

        parameters = HMCParameters(
            np.ones(initial_state.position.shape[0], dtype=np.int32)
            * num_integration_steps,
            step_size,
            inverse_mass_matrix,
        )

        return last_chain_state, parameters, warmup_chain

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

    def make_trace(
        self,
        chain: Tuple[HMCState, HMCInfo],
        unravel_fn: Callable,
    ) -> Tuple[Dict, Dict]:
        """Translate the raw chain to a format that `Trace` understands.

        Parameters
        ----------
        chain
            A tuple that contains the HMC sampler's states and additional information
            on the sampling process.
        unravel_fn
            A functions that returns flattened variables to their original shape.

        Returns
        -------
        A dictionary with the variables' samples and a dictionary with sampling
        information, which are later processed to build a Trace object.

        """
        state, info = chain

        # We ravelled positions before sampling to be able to make
        # computations on flat arrays in the backend. We now need to bring
        # their to their original shape before adding to the trace.
        unravel_chain = jax.vmap(unravel_fn, in_axes=(0,))
        try:
            samples = jax.vmap(unravel_chain, in_axes=(1,))(state.position)
        except IndexError:  # unravel single samples
            samples = unravel_chain(state.position)

        # I added all these `type: ignore` because I am getting lazy about types.
        # The chain that we accumulate is not a list of HMC State, but rather
        # and HMC State where fields are list of the subequent values in the
        # chain. This is due to the way JAX handles "pytrees". Thus at this
        # point we are not manipulating HMCStates but a ChainState that looks
        # like a HMCState but with arrays as fields.
        # We can either create new states (verbose but preferred, I am not sure
        # this distinction is clear for newcomers) or keep silencing the checker.
        # Please keep this comment if the later is chosen.
        # TODO: Create new HMCChain type.
        sampling_info = {
            "potential_energy": state.potential_energy.T,  # type: ignore
            "acceptance_probability": info.acceptance_probability.T,  # type: ignore
            "is_divergent": info.is_divergent.T,  # type: ignore
            "energy": info.energy.T,  # type: ignore
            "step_size": info.proposal_info.step_size.T,  # type: ignore
            "num_integration_steps": info.proposal_info.num_integration_steps.T,  # type: ignore
        }

        return samples, sampling_info

    def make_warmup_trace(
        self,
        chain: Tuple[HMCState, StanWarmupState, HMCInfo],
        unravel_fn: Callable,
    ) -> Tuple[Dict, Dict, Dict]:
        """Translate the Warmup chains to a format `Trace` can understand.

        Parameters
        ----------
        chain
            A tuple that contains the HMC sampler's states and additional information
            on the sampling and warmup process.
        unravel_fn
            A functions that returns flattened variables to their original shape.

        Returns
        -------
        Three dictionaries. The first contains the variables' samples, the second
        information about the sampling process and the third information about
        the consecutive states of the warmup algorithms.

        """
        chain_state, warmup_info, chain_info = chain
        samples, sampling_info = self.make_trace((chain_state, chain_info), unravel_fn)

        warmup_info_dict = {
            "log_step_size": warmup_info.da_state.log_step_size,
            "log_step_size_avg": warmup_info.da_state.log_step_size_avg,
            "inverse_mass_matrix": warmup_info.mm_state.inverse_mass_matrix,
        }

        return samples, sampling_info, warmup_info_dict

    def _to_potential(self, loglikelihood: Callable) -> Callable:
        """The potential in the Hamiltonian Monte Carlo algorithm is equal to
        minus the log-likelihood.

        """

        def potential(array: np.DeviceArray) -> float:
            return -loglikelihood(array)

        return potential
