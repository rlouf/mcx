"""Generate dynamics on a Euclidean Manifold or Riemannian Manifold.

.. note:
    This file is a "flat zone": positions and logprobs are 1 dimensional
    arrays. Raveling and unraveling logic must happen outside.
"""
from typing import Any, Callable, Tuple

import jax
from jax import numpy as np
from jax.numpy import DeviceArray as Array


__all__ = ["gaussian_euclidean_metric"]


MetricFactory = Callable[
    [Any], Tuple[Callable[[jax.random.PRNGKey], Array], Callable[[Array], float]]
]


def gaussian_euclidean_metric(
    mass_matrix_sqrt: Array, inverse_mass_matrix: Array
) -> Tuple[Callable, Callable]:
    """Emulate dynamics on an Euclidean Manifold [1]_ for vanilla Hamiltonian
    Monte Carlo with a standard gaussian as the conditional density
    :math:`\\pi(momentum|position)`.

    References
    ----------
    .. [1]: Betancourt, Michael. "A general metric for Riemannian manifold
            Hamiltonian Monte Carlo." International Conference on Geometric Science of
            Information. Springer, Berlin, Heidelberg, 2013.
    """

    shape = np.shape(mass_matrix_sqrt)[:1]

    def momentum_generator(rng_key: jax.random.PRNGKey) -> Array:
        std = jax.random.normal(rng_key, shape)
        if np.ndim(mass_matrix_sqrt) == 1:
            return np.multiply(std, mass_matrix_sqrt)
        if np.ndim(mass_matrix_sqrt) == 2:
            return np.dot(std, mass_matrix_sqrt)
        else:
            raise ValueError(
                "The mass matrix has the wrong number of dimensions: "
                + "expected 1 or 2, got {}.".format(np.ndim(mass_matrix_sqrt))
            )

    def kinetic_energy(momentum: Array) -> float:
        if np.dim(inverse_mass_matrix) == 1:
            v = np.multiply(inverse_mass_matrix, momentum)
        elif np.dim(inverse_mass_matrix.ndim) == 2:
            v = np.matmul(inverse_mass_matrix, momentum)
        else:
            raise ValueError(
                "The inverse mass matrix has the wrong number of dimensions: "
                + "expected 1 or 2, got {}.".format(np.ndim(inverse_mass_matrix))
            )

        return 0.5 * np.dot(v, momentum)

    return momentum_generator, kinetic_energy
