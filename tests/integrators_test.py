import jax
from jax import numpy as np
import pytest

from mcx.inference.integrators import (
    IntegratorState,
    four_stages_integrator,
    mclachlan_integrator,
    velocity_verlet,
    yoshida_integrator,
)


def HarmonicOscillator(inverse_mass_matrix, k=5, m=1.0):
    """Potential and Kinetic energy of an harmonic oscillator.
    """

    def potential_energy(x):
        return np.sum(0.5 * k * np.square(x))

    def kinetic_energy(p):
        v = np.multiply(inverse_mass_matrix, p)
        return np.sum(0.5 * np.dot(v, p))

    return potential_energy, kinetic_energy


def FreeFall(inverse_mass_matrix, g=9.81, m=1.0):
    """Potential and kinetic energy of a free-falling object.
    """

    def potential_energy(h):
        return np.sum(m * g * h)

    def kinetic_energy(p):
        v = np.multiply(inverse_mass_matrix, p)
        return np.sum(0.5 * np.dot(v, p))

    return potential_energy, kinetic_energy


integration_examples = [
    {
        "model": HarmonicOscillator,
        "num_step": 100,
        "step_size": 0.01,
        "q": 0.0,
        "p": 1.0,
        "inverse_mass_matrix": np.array([1.0]),
    },
    {
        "model": FreeFall,
        "num_step": 100,
        "step_size": 0.01,
        "q": 0.0,
        "p": 1.0,
        "inverse_mass_matrix": np.array([1.0]),
    },
]

integrator_steps = [
    yoshida_integrator,
    velocity_verlet,
    four_stages_integrator,
    mclachlan_integrator,
]


@pytest.mark.parametrize("example", integration_examples)
@pytest.mark.parametrize("integrator_step", integrator_steps)
def test_velocity_verlet(example, integrator_step):
    model = example["model"]
    potential, kinetic_energy = model(example["inverse_mass_matrix"])
    step = integrator_step(potential, kinetic_energy)
    step_size = example["step_size"]

    q = example["q"]
    p = example["p"]
    initial_state = IntegratorState(q, p, potential(q), jax.grad(potential)(q))
    final_state = jax.lax.fori_loop(
        0, example["num_step"], lambda i, state: step(state, step_size), initial_state
    )

    # Symplectic integrators conserve energy
    energy = potential(q) + kinetic_energy(p)
    new_energy = potential(final_state.position) + kinetic_energy(final_state.momentum)
    print(energy, new_energy.item())
    assert energy == pytest.approx(new_energy.item(), 1e-4)
