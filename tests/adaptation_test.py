import jax
from jax import numpy as np

from mcx.inference.integrators import velocity_verlet
from mcx.inference.kernels import HMCState, hmc_kernel
from mcx.inference.metrics import gaussian_euclidean_metric
from mcx.inference.proposals import hmc_proposal
from mcx.inference.warmup.step_size_adaptation import find_reasonable_step_size


def test_find_reasonable_step_size():
    def potential_fn(x):
        return np.sum(0.5 * np.square(x))

    rng_key = jax.random.PRNGKey(0)

    inv_mass_matrix = np.array([1.0])

    init_position = np.array([3.0])

    potential_energy, potential_energy_grad = jax.value_and_grad(potential_fn)(
        init_position
    )
    init_state = HMCState(init_position, potential_energy, potential_energy_grad)

    def kernel_generator(step_size, inv_mass_matrix):
        momentum_generator, kinetic_energy = gaussian_euclidean_metric(inv_mass_matrix)
        integrator_step = velocity_verlet(potential_fn, kinetic_energy)
        proposal = hmc_proposal(integrator_step, step_size, 1)
        kernel = hmc_kernel(proposal, momentum_generator, kinetic_energy, potential_fn)
        return kernel

    # Test that the algorithm actually does something
    epsilon_1 = find_reasonable_step_size(
        rng_key,
        kernel_generator,
        init_state,
        inv_mass_matrix,
        1.0,
        0.95,
    )
    assert epsilon_1 != 1.0

    # Different target acceptance rate
    epsilon_3 = find_reasonable_step_size(
        rng_key,
        kernel_generator,
        init_state,
        inv_mass_matrix,
        1.0,
        0.05,
    )
    assert epsilon_3 != epsilon_1
