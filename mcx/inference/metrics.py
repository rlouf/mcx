"""Generate dynamics on a Euclidean Manifold or Riemannian Manifold.

.. note:
    This file is a "flat zone": positions and logprobs are 1 dimensional
    arrays. Raveling and unraveling logic must happen outside.
"""
from typing import Callable, Tuple

import jax
import jax.numpy as np
from jax.numpy import DeviceArray as Array


__all__ = ["gaussian_euclidean_metric"]


KineticEnergy = Callable[[Array], float]
MomentumGenerator = Callable[[jax.random.PRNGKey], Array]


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

    if np.ndim(inverse_mass_matrix) != np.ndim(mass_matrix_sqrt):
        raise ValueError(
            "The inverse mass matrix and mass matrix have a different "
            "number of dimensions: {} vs {} respectively.".format(
                np.ndim(inverse_mass_matrix), np.dim(mass_matrix_sqrt)
            )
        )
    ndim = np.ndim(mass_matrix_sqrt)

    if np.shape(inverse_mass_matrix) != np.shape(mass_matrix_sqrt):
        raise ValueError(
            "The inverse mass matrix and mass matrix have different "
            "shapes: {} vs {} respectively.".format(
                np.ndim(inverse_mass_matrix), np.dim(mass_matrix_sqrt)
            )
        )
    shape = np.shape(mass_matrix_sqrt)[:1]

    if ndim == 1:  # diagonal mass matrix

        @jax.jit
        def momentum_generator(rng_key: jax.random.PRNGKey) -> Array:
            std = jax.random.normal(rng_key, shape)
            p = np.multiply(std, mass_matrix_sqrt)
            return p

        @jax.jit
        def kinetic_energy(momentum: Array) -> float:
            velocity = np.multiply(inverse_mass_matrix, momentum)
            return 0.5 * np.dot(velocity, momentum)

        return momentum_generator, kinetic_energy

    elif ndim == 2:

        @jax.jit
        def momentum_generator(rng_key: jax.random.PRNGKey) -> Array:
            std = jax.random.normal(rng_key, shape)
            p = np.dot(std, mass_matrix_sqrt)
            return p

        @jax.jit
        def kinetic_energy(momentum: Array) -> float:
            velocity = np.matmul(inverse_mass_matrix, momentum)
            return 0.5 * np.dot(velocity, momentum)

        return momentum_generator, kinetic_energy

    else:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions: "
            + "expected 1 or 2, got {}.".format(np.ndim(inverse_mass_matrix))
        )
