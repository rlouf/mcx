"""Adaptive algorithms for Markov Chain Monte Carlo.

This is a collection of re-usable adaptive schemes for monte carlo algorithms.
The algorithms are used during the warm-up phase of the inference and are
decorrelated from any particular algorithm (dynamic HMC's adaptive choice of
path length is not included, for instance).

The Stan Manual [1]_ is a very good reference on automatic tuning of
parameters used in Hamiltonian Monte Carlo.

.. [1]: "HMC Algorithm Parameters", Stan Manual
        https://mc-stan.org/docs/2_20/reference-manual/hmc-algorithm-parameters.html
"""
from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
from jax import numpy as np


__all__ = ["mass_matrix_adaptation"]


class WelfordAlgorithmState(NamedTuple):
    mean: float  # the current sample mean
    m2: float  # the current value of the sum of difference of squares
    sample_size: int  # the sample size


class MassMatrixAdaptationState(NamedTuple):
    inverse_mass_matrix: np.DeviceArray
    wc_state: WelfordAlgorithmState


def mass_matrix_adaptation(
    is_diagonal_matrix: bool = True,
) -> Tuple[Callable, Callable, Callable]:
    """Adapts the values in the mass matrix by computing the covariance
    between parameters.
    """
    wc_init, wc_update, wc_final = welford_algorithm(is_diagonal_matrix)

    def init(n_dims: int) -> MassMatrixAdaptationState:
        if is_diagonal_matrix:
            inverse_mass_matrix = np.ones(n_dims)
        else:
            inverse_mass_matrix = np.identity(n_dims)

        wc_state = wc_init(n_dims)

        return MassMatrixAdaptationState(inverse_mass_matrix, wc_state)

    @jax.jit
    def update(
        state: MassMatrixAdaptationState, value: np.DeviceArray
    ) -> MassMatrixAdaptationState:
        inverse_mass_matrix, wc_state = state
        wc_state = wc_update(wc_state, value)
        return MassMatrixAdaptationState(inverse_mass_matrix, wc_state)

    @jax.jit
    def final(state: MassMatrixAdaptationState) -> MassMatrixAdaptationState:
        _, wc_state = state
        covariance, count, mean = wc_final(wc_state)

        # Regularize the covariance matrix, see Stan
        scaled_covariance = (count / (count + 5)) * covariance
        shrinkage = 1e-3 * (5 / (count + 5))
        if is_diagonal_matrix:
            inverse_mass_matrix = scaled_covariance + shrinkage
        else:
            inverse_mass_matrix = scaled_covariance + shrinkage * np.identity(
                mean.shape[0]
            )

        ndims = np.shape(inverse_mass_matrix)[-1]
        new_mm_state = MassMatrixAdaptationState(inverse_mass_matrix, wc_init(ndims))

        return new_mm_state

    return init, update, final


def welford_algorithm(is_diagonal_matrix: bool) -> Tuple[Callable, Callable, Callable]:
    """Welford's online estimator of covariance.

    It is possible to compute the variance of a population of values in an
    on-line fashion to avoid storing intermediate results. The naive recurrence
    relations between the sample mean and variance at a step and the next are
    however not numerically stable.

    Welford's algorithm uses the sum of square of differences
    :math:`M_{2,n} = \\sum_{i=1}^n \\left(x_i-\\overline{x_n}\right)^2`
    for updating where :math:`x_n` is the current mean and the following
    recurrence relationships

    .. math:
        M_{2,n} = M_{2, n-1} + (x_n-\\overline{x}_{n-1})(x_n-\\overline{x}_n)
        \\sigma_n^2 = \\frac{M_{2,n}}{n}
    """

    def init(n_dims: int) -> WelfordAlgorithmState:
        """Initialize the covariance estimation.

        When the matrix is diagonal it is sufficient to work with an array that contains
        the diagonal value. Otherwise we need to work with the matrix in full.

        Argument
        --------
        n_dims: int
            The number of dimensions of the problem, which corresponds to the size
            of the corresponding square mass matrix.
        """
        count = 0
        mean = np.zeros(n_dims)
        if is_diagonal_matrix:
            m2 = np.zeros(n_dims)
        else:
            m2 = np.zeros((n_dims, n_dims))
        return WelfordAlgorithmState(mean, m2, count)

    @jax.jit
    def update(
        state: WelfordAlgorithmState, value: np.DeviceArray
    ) -> WelfordAlgorithmState:
        """Update the M2 matrix using the new value.

        Arguments:
        ----------
        state:
            The current state of the Welford Algorithm
        value: jax.numpy.DeviceArray, shape (1,)
            The new sample used to update m2
        """
        mean, m2, count = state
        count = count + 1

        delta = value - mean
        mean = mean + delta / count
        updated_delta = value - mean
        if is_diagonal_matrix:
            new_m2 = m2 + delta * updated_delta
        else:
            new_m2 = m2 + np.outer(delta, updated_delta)

        return WelfordAlgorithmState(mean, new_m2, count)

    def covariance(state: WelfordAlgorithmState) -> np.DeviceArray:
        mean, m2, count = state
        covariance = m2 / (count - 1)
        return covariance, count, mean

    return init, update, covariance
