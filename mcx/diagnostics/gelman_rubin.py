"""Kernel to compute the Gelman-Rubin convergence diagnostic (Rhat) online.
"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp


class WelfordAlgorithmState(NamedTuple):
    """State carried through the Welford algorithm.

    mean
        The running sample mean.
    m2
        The running value of the sum of difference of squares. See documentation
        of the `welford_algorithm` function for an explanation.
    sample_size
        The number of successive states the previous values have been computed on;
        also the current number of iterations of the algorithm.
    """

    mean: float
    m2: float
    sample_size: int


class GelmanRubinState(NamedTuple):
    w_state: WelfordAlgorithmState
    rhat: jnp.DeviceArray
    metric: jnp.DeviceArray
    metric_name: str


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

    Parameters
    ----------
    is_diagonal_matrix
        When True the algorithm adapts and returns a diagonal mass matrix
        (default), otherwise adapts and returns a dense mass matrix.

    .. math:
        M_{2,n} = M_{2, n-1} + (x_n-\\overline{x}_{n-1})(x_n-\\overline{x}_n)
        \\sigma_n^2 = \\frac{M_{2,n}}{n}
    """

    def init(n_chains: int, n_dims: int) -> WelfordAlgorithmState:
        """Initialize the covariance estimation.

        When the matrix is diagonal it is sufficient to work with an array that contains
        the diagonal value. Otherwise we need to work with the matrix in full.

        Parameters
        ----------
        n_chains: int
            The number of chains being run
        n_dims: int
            The number of variables
        """
        sample_size = 0
        mean = jnp.zeros((n_chains, n_dims))
        if is_diagonal_matrix:
            m2 = jnp.zeros((n_chains, n_dims))
        else:
            m2 = jnp.zeros((n_chains, n_chains, n_dims))
        return WelfordAlgorithmState(mean, m2, sample_size)

    @jax.jit
    def update(
        state: WelfordAlgorithmState, value: jnp.DeviceArray
    ) -> WelfordAlgorithmState:
        """Update the M2 matrix using the new value.

        Parameters
        ----------
        state: WelfordAlgorithmState
            The current state of the Welford Algorithm
        value: jax.numpy.DeviceArray, shape (1,)
            The new sample (typically position of the chain) used to update m2
        """
        mean, m2, sample_size = state
        sample_size = sample_size + 1

        delta = value - mean
        mean = mean + delta / sample_size
        updated_delta = value - mean
        if is_diagonal_matrix:
            new_m2 = m2 + delta * updated_delta
        else:
            new_m2 = m2 + jnp.outer(updated_delta, delta)

        return WelfordAlgorithmState(mean, new_m2, sample_size)

    def covariance(
        state: WelfordAlgorithmState,
    ) -> Tuple[jnp.DeviceArray, int, jnp.DeviceArray]:
        mean, m2, sample_size = state
        covariance = m2 / (sample_size - 1)
        return covariance, sample_size, mean

    return init, update, covariance


def online_gelman_rubin():
    """Online estimation of the Gelman-Rubin diagnostic."""

    w_init, w_update, w_covariance = welford_algorithm(True)

    def init(init_state):
        """Initialise the online gelman/rubin estimator

        Parameters
        ----------
        num_chains: int
            The number of chains being run

        Returns
        -------
        GelmanRubinState with all values set to zeros.

        """
        n_chains, n_dims = init_state.position.shape
        w_state = w_init(n_chains, n_dims)
        return GelmanRubinState(w_state, 0, jnp.nan, "worst_rhat")

    def update(chain_state, rhat_state):
        """Update rhat estimates

        Parameters
        ----------
        chain_state: HMCState
            The chain state
        rhat_state: GelmanRubinState
            The GelmanRubinState from the previous draw

        Returns
        -------
        An updated GelmanRubinState object
        """
        within_state, _, _, metric_name = rhat_state

        positions = chain_state.position
        within_state = w_update(within_state, positions)
        covariance, step, mean = w_covariance(within_state)
        within_var = jnp.mean(covariance, axis=0)
        between_var = jnp.var(mean, axis=0, ddof=1)
        estimator = ((step - 1) / step) * within_var + between_var
        rhat = jnp.sqrt(estimator / within_var)
        worst_rhat = rhat[jnp.argmax(jnp.abs(rhat - 1.0))]

        return GelmanRubinState(within_state, rhat, worst_rhat, metric_name)

    return init, update


def split_gelman_rubin():
    pass
