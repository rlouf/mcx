"""Adaptive algorithms for Markov Chain Monte Carlo.

This is a collection of re-usable adaptive schemes for monte carlo algorithms.
It only contains algorithms that are used during the warm-up phase of the
inference and are decorrelated from any particular kernel (NUTS' adaptive
choice of path length is not included, for instance).

The Stan Manual [1]_ is a very good reference on automatic tuning of
parameters used in Hamiltonian Monte Carlo.

.. note:
    This is a "flat zone": values used to update the step size or the mass
    matrix are 1D arrays. Raveling/unraveling logic should happen at a higher
    level.

.. [1]: "HMC Algorithm Parameters", Stan Manual
        https://mc-stan.org/docs/2_20/reference-manual/hmc-algorithm-parameters.html
"""
from typing import Callable, NamedTuple, Tuple

from jax import numpy as np
from jax import scipy
from jax.numpy import DeviceArray as Array


__all__ = ["dual_averaging", "find_reasonable_step_size", "mass_matrix_adaptation"]


class DualAveragingState(NamedTuple):
    log_step_size: float
    log_step_size_avg: float
    t: int
    avg_error: float
    mu: float


class WelfordAlgorithmState(NamedTuple):
    mean: float  # the current sample mean
    m2: float  # the current value of the sum of difference of squares
    count: int  # the sample size


class MassMatrixAdaptationState(NamedTuple):
    inverse_mass_matrix: Array
    mass_matrix_sqrt: Array
    wc_state: WelfordAlgorithmState


def dual_averaging(
    t0: int = 10, gamma: float = 0.05, kappa: float = 0.75, target: float = 0.65
) -> Tuple[Callable, Callable]:
    """Tune the step size to achieve a desired target acceptance rate.

    Let us note :math:`\\epsilon` the current step size, :math:`\\alpha_t` the
    metropolis acceptance rate at time :math:`t` and :math:`\\delta` the desired
    aceptance rate. We define:

    .. math:
        H_t = \\delta - \\alpha_t

    the error at time t. We would like to find a procedure that adapts the
    value of :math:`\\epsilon` such that :math:`h(x) =\\mathbb{E}\\left[H_t|\\epsilon\\right] = 0`

    Following [1]_, the authors of [2]_ proposed the following update scheme. If
    we note :math:``x = \\log \\epsilon` we follow:

    .. math:
        x_{t+1} \\LongLeftArrow \\mu - \\frac{\\sqrt{t}}{\\gamma} \\frac{1}{t+t_0} \\sum_{i=1}^t H_i
        \\overline{x}_{t+1} \\LongLeftArrow x_{t+1}\\, t^{-\\kappa}  + \\left(1-t^\\kappa\\right)\\overline{x}_t

    :math:`\\overline{x}_{t}` is guaranteed to converge to a value such that
    :math:`h(\\overline{x}_t)` converges to 0, i.e. the metropolis acceptance
    rate converges to the desired rate.

    See reference [2]_ (section 3.2.1) for a detailed discussion.

    Arguments
    ---------
    t0: float > 0
        Free parameter that stabilizies the initial iterations of the algorithm.
        Large values may slow down convergence. Introduced in [2]_ with a default
        value of 10.
    gamma: float
        Controls the speed of convergence of the scheme. The authors of [2]_ recommend
        a value of 0.05.
    kappa: float in ]0.5, 1]
        Controls the weights of past steps in the current update. The scheme will
        quickly forget earlier step for a small value of `kappa`. Introduced
        in [2]_, with a recommended value of .75

    Returns
    -------
    init: function
        Function to initialize the state of the dual averaging scheme.
    update: function
        Function that updates the state of the dual averaging scheme.

    References
    ----------

    .. [1] Nesterov, Yurii. "Primal-dual subgradient methods for convex
           problems." Mathematical programming 120.1 (2009): 221-259.
    .. [2] Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler:
           adaptively setting path lengths in Hamiltonian Monte Carlo." Journal
           of Machine Learning Research 15.1 (2014): 1593-1623.
    """

    def init(inital_step_size: float) -> DualAveragingState:
        """Initialize the state of the dual averaging scheme.

        The current state of the dual averaging scheme consists, following [2]_,
        of the following quantities:

        - avg_error: The current time-average value of :math:`H_t` defined above;
        - log_step: The logarithm of the current step size;
        - avg_log_step: The logarithm of the current time-weighted average of the step size.

        The parameter :math:`\\mu` is set to :math:`\\log(10 \\epsilon_1)`
        where :math:`\\epsilon_1` is the initial value of the step size.
        """
        mu: float = np.log(10 * inital_step_size)
        t = t0
        avg_error: float = 0
        log_step_size: float = 0
        log_step_size_avg: float = 0
        return DualAveragingState(log_step_size, log_step_size_avg, t, avg_error, mu)

    def update(p_accept: float, state: DualAveragingState) -> DualAveragingState:
        """Update the state of the Dual Averaging adaptive scheme.

        Arguments:
        ----------
        p_accept: float in [0, 1]
            The current metropolis acceptance rate.
        state: tuple
            The current state of the dual averaging scheme.
        """
        log_step, avg_log_step, t, avg_error, mu = state
        eta_t = t ** (-kappa)
        avg_error = (1 - (1 / t)) * avg_error + (target - p_accept) / t
        log_step_size = mu - (np.sqrt(t) / gamma) * avg_error
        log_step_size_avg = eta_t * log_step + (1 - eta_t) * avg_log_step
        return DualAveragingState(log_step_size, log_step_size_avg, t, avg_error, mu)

    return init, update


def mass_matrix_adaptation(
    is_diagonal_matrix: bool = True,
) -> Tuple[Callable, Callable, Callable]:
    wc_init, wc_update, wc_final = welford_algorithm(is_diagonal_matrix)

    def init(n_dims: int) -> MassMatrixAdaptationState:
        if is_diagonal_matrix:
            inverse_mass_matrix = np.ones(n_dims)
        else:
            inverse_mass_matrix = np.identity(n_dims)
        mass_matrix_sqrt = inverse_mass_matrix

        wc_state = wc_init(n_dims)

        return MassMatrixAdaptationState(
            inverse_mass_matrix, mass_matrix_sqrt, wc_state
        )

    def update(
        state: MassMatrixAdaptationState, value: Array
    ) -> MassMatrixAdaptationState:
        mass_matrix_sqrt, inverse_mass_matrix, wc_state = state
        wc_state = wc_update(wc_state, value)
        return MassMatrixAdaptationState(
            mass_matrix_sqrt, inverse_mass_matrix, wc_state
        )

    def final(state: MassMatrixAdaptationState,) -> Tuple[Array, Array]:
        mass_matrix_sqrt, inverse_mass_matrix, wc_state = state
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

        if np.ndim(inverse_mass_matrix) == 2:
            mass_matrix_sqrt = cholesky_triangular(inverse_mass_matrix)
        else:
            mass_matrix_sqrt = np.sqrt(np.reciprocal(inverse_mass_matrix))

        return inverse_mass_matrix, mass_matrix_sqrt

    return init, update, mass_matrix


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

    def update(state: WelfordAlgorithmState, value: Array) -> WelfordAlgorithmState:
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
            m2 = m2 + delta * updated_delta
        else:
            m2 = m2 + np.outer(delta, updated_delta)

        return WelfordAlgorithmState(mean, m2, count)

    def covariance(state: WelfordAlgorithmState) -> np.DeviceArray:
        mean, m2, count = state
        covariance = m2 / (count - 1)
        return covariance, count, mean

    return init, update, covariance


# Sourced from numpyro.distributions.utils.py
# Copyright Contributors to the NumPyro project.
# SPDX-License-Identifier: Apache-2.0
def cholesky_triangular(matrix: Array) -> Array:
    tril_inv = np.swapaxes(
        np.linalg.cholesky(matrix[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
    )
    identity = np.broadcast_to(np.identity(matrix.shape[-1]), tril_inv.shape)
    return scipy.linalg.solve_triangular(tril_inv, identity, lower=True)
