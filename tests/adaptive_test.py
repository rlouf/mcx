import jax
from jax import numpy as np
from jax.flatten_util import ravel_pytree

from mcx.inference.adaptive import find_reasonable_step_size
from mcx.inference.integrators import velocity_verlet
from mcx.inference.kernels import hmc_init
from mcx.inference.metrics import gaussian_euclidean_metric


def test_find_reasonable_step_size():
    def potential_fn(x):
        return np.sum(0.5 * np.square(x))

    rng_key = jax.random.PRNGKey(0)

    init_position = np.array([3.0])
    inv_mass_matrix = np.array([1.0])
    mass_matrix_sqrt = np.array([1.0])

    init_state = hmc_init(init_position, potential_fn)
    _, unravel_fn = ravel_pytree(init_position)
    momentum_generator, kinetic_energy = gaussian_euclidean_metric(
        mass_matrix_sqrt, inv_mass_matrix, unravel_fn
    )
    integrator_step = velocity_verlet(potential_fn, kinetic_energy)

    # Test that the algorithm actually does something
    epsilon_1 = find_reasonable_step_size(
        rng_key,
        momentum_generator,
        kinetic_energy,
        potential_fn,
        integrator_step,
        init_state,
        1.0,
        0.95,
    )
    assert epsilon_1 != 1.

    # Different target acceptance rate
    epsilon_3 = find_reasonable_step_size(
        rng_key,
        momentum_generator,
        kinetic_energy,
        potential_fn,
        integrator_step,
        init_state,
        1.0,
        0.05,
    )
    assert epsilon_3 != epsilon_1
