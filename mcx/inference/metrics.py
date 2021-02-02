"""Generate dynamics on a Euclidean Manifold or Riemannian Manifold.
"""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as scipy

__all__ = ["gaussian_euclidean_metric"]


KineticEnergy = Callable[[jnp.DeviceArray], float]
MomentumGenerator = Callable[[jax.random.PRNGKey], jnp.DeviceArray]


def gaussian_euclidean_metric(
    inverse_mass_matrix: jnp.DeviceArray,
) -> Tuple[Callable, Callable]:
    """Emulate dynamics on an Euclidean Manifold [1]_ for vanilla Hamiltonian
    Monte Carlo with a standard gaussian as the conditional probability density
    :math:`\\pi(momentum|position)`.

    References
    ----------
    .. [1]: Betancourt, Michael. "A general metric for Riemannian manifold
            Hamiltonian Monte Carlo." International Conference on Geometric Science of
            Information. Springer, Berlin, Heidelberg, 2013.
    """

    ndim = jnp.ndim(inverse_mass_matrix)
    shape = jnp.shape(inverse_mass_matrix)[:1]

    if ndim == 1:  # diagonal mass matrix

        mass_matrix_sqrt = jnp.sqrt(jnp.reciprocal(inverse_mass_matrix))

        @jax.jit
        def momentum_generator(rng_key: jax.random.PRNGKey) -> jnp.DeviceArray:
            std = jax.random.normal(rng_key, shape)
            p = jnp.multiply(std, mass_matrix_sqrt)
            return p

        @jax.jit
        def kinetic_energy(momentum: jnp.DeviceArray) -> float:
            velocity = jnp.multiply(inverse_mass_matrix, momentum)
            return 0.5 * jnp.dot(velocity, momentum)

        return momentum_generator, kinetic_energy

    elif ndim == 2:

        mass_matrix_sqrt = cholesky_of_inverse(inverse_mass_matrix)

        @jax.jit
        def momentum_generator(rng_key: jax.random.PRNGKey) -> jnp.DeviceArray:
            std = jax.random.normal(rng_key, shape)
            p = jnp.dot(std, mass_matrix_sqrt)
            return p

        @jax.jit
        def kinetic_energy(momentum: jnp.DeviceArray) -> float:
            velocity = jnp.matmul(inverse_mass_matrix, momentum)
            return 0.5 * jnp.dot(velocity, momentum)

        return momentum_generator, kinetic_energy

    else:
        raise ValueError(
            "The mass matrix has the wrong number of dimensions:"
            f" expected 1 or 2, got {jnp.dim(inverse_mass_matrix)}."
        )


# Sourced from numpyro.distributions.utils.py
# Copyright Contributors to the NumPyro project.
# SPDX-License-Identifier: Apache-2.0
def cholesky_of_inverse(matrix):
    # This formulation only takes the inverse of a triangular matrix
    # which is more numerically stable.
    # Refer to:
    # https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    tril_inv = jnp.swapaxes(
        jnp.linalg.cholesky(matrix[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
    )
    identity = jnp.broadcast_to(jnp.identity(matrix.shape[-1]), tril_inv.shape)
    return scipy.linalg.solve_triangular(tril_inv, identity, lower=True)
