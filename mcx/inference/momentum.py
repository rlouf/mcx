from typing import Callable, Tuple

from jax import numpy as np
from jax import random

__all__ = ["euclidean_manifold_dynamics"]


def euclidean_manifold_dynamics() -> Tuple[Callable, Callable]:
    """Emulate dynamics on an Euclidean Manifold for vanilla Hamiltonian
    Monte Carlo.

    References
    ----------
    .. [1]: Betancourt, Michael. "A general metric for Riemannian manifold
            Hamiltonian Monte Carlo." International Conference on Geometric Science of
            Information. Springer, Berlin, Heidelberg, 2013.
    """

    def momentum_generator(
        rng_key: random.PRNGKey, mass_matrix_sqrt: np.DeviceArray
    ) -> np.DeviceArray:
        shape = np.shape(mass_matrix_sqrt)[:1]
        std = random.normal(rng_key, shape)
        if mass_matrix_sqrt.ndim == 1:
            return np.multiply(std, mass_matrix_sqrt)
        if mass_matrix_sqrt.ndim == 2:
            return np.dot(std, mass_matrix_sqrt)
        else:
            raise ValueError(
                "The mass matrix has the wrong number of shapes: "
                + "expected 1 or 2, got {}.".format(mass_matrix_sqrt.ndim)
            )

    def kinetic_energy(
        p: np.DeviceArray, inverse_mass_matrix: np.DeviceArray
    ) -> np.DeviceArray:
        if inverse_mass_matrix.ndim == 1:
            v = np.matmul(inverse_mass_matrix, p)
        elif inverse_mass_matrix.ndim == 2:
            v = np.dot(inverse_mass_matrix, p)

        return 0.5 * np.dot(v, p)

    return momentum_generator, kinetic_energy
