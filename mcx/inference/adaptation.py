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

from mcx.inference.integrators import hmc_proposal
from mcx.inference.kernels import hmc_kernel, HMCState


__all__ = [
    "dual_averaging",
    "find_reasonable_step_size",
    "mass_matrix_adaptation",
    "longest_batch_before_turn",
]


# --------------------------------------
#      == STEP SIZE ADAPTATION ==
# --------------------------------------


class DualAveragingState(NamedTuple):
    log_step_size: float
    log_step_size_avg: float
    t: int
    avg_error: float
    mu: float


def dual_averaging(
    t0: int = 10, gamma: float = 0.05, kappa: float = 0.75, target: float = 0.65
) -> Tuple[Callable, Callable]:
    """Tune the step size in order to achieve a desired target acceptance rate.

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
    :math:`h(\\overline{x}_t)` converges to 0, i.e. the Metropolis acceptance
    rate converges to the desired rate.

    See reference [2]_ (section 3.2.1) for a detailed discussion.

    Arguments
    ---------
    t0: float > 0
        Free parameter that stabilizes the initial iterations of the algorithm.
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
    init:
        A function that initializes the state of the dual averaging scheme.
    update: function
        A function that updates the state of the dual averaging scheme.

    References
    ----------

    .. [1]: Nesterov, Yurii. "Primal-dual subgradient methods for convex
            problems." Mathematical programming 120.1 (2009): 221-259.
    .. [2]: Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler:
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

    @jax.jit
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


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def find_reasonable_step_size(
    rng_key: jax.random.PRNGKey,
    momentum_generator: Callable,
    kinetic_energy: Callable,
    potential_fn: Callable,
    integrator_step: Callable,
    reference_hmc_state: HMCState,
    initial_step_size: float = 1.0,
    target_accept: float = 0.65,
) -> float:
    """Find a reasonable initial step size during warmup.

    While the dual averaging scheme is guaranteed to converge to a reasonable
    value for the step size starting from any value, choosing a good first
    value can speed up the convergence. This heuristics doubles and halves the
    step size until the acceptance probability of the HMC proposal crosses the
    .5 value.

    Returns
    -------
    float
        A reasonable first value for the step size.

    Reference
    ---------
    .. [1]: Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler:
            adaptively setting path lengths in Hamiltonian Monte Carlo." Journal
            of Machine Learning Research 15.1 (2014): 1593-1623.
    """
    fp_limit = np.finfo(jax.lax.dtype(initial_step_size))

    def _new_hmc_kernel(step_size: float) -> Callable:
        """Return a HMC kernel that operates with the provided step size."""
        proposal_generator = hmc_proposal(integrator_step, step_size, 1)
        kernel = hmc_kernel(
            proposal_generator, momentum_generator, kinetic_energy, potential_fn
        )
        return kernel

    def _update(search_state: Tuple) -> Tuple:
        rng_key, direction, _, step_size = search_state
        _, rng_key = jax.random.split(rng_key)

        step_size = (2.0 ** direction) * step_size
        kernel = _new_hmc_kernel(step_size)
        _, hmc_info = kernel(rng_key, reference_hmc_state)

        new_direction = np.where(target_accept < hmc_info.acceptance_probability, 1, -1)
        return (rng_key, new_direction, direction, step_size)

    def _do_continue(search_state: Tuple) -> bool:
        """Decides whether the search should continue.

        The search stops when it crosses the `target_accept` threshold, i.e.
        when the current direction is opposite to the previous direction.
        """
        _, direction, previous_direction, step_size = search_state

        not_too_large = (step_size < fp_limit.max) | (direction <= 0)
        not_too_small = (step_size > fp_limit.tiny) | (direction >= 0)
        is_step_size_not_extreme = not_too_large & not_too_small
        has_acceptance_rate_not_crossed_threshold = (previous_direction == 0) | (
            direction == previous_direction
        )
        return is_step_size_not_extreme & has_acceptance_rate_not_crossed_threshold

    _, _, _, step_size = jax.lax.while_loop(
        _do_continue, _update, (rng_key, 0, 0, initial_step_size)
    )
    return step_size


# --------------------------------------
#      == MASS MATRIX ADAPTATION ==
# --------------------------------------


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

    def update(
        state: MassMatrixAdaptationState, value: np.DeviceArray
    ) -> MassMatrixAdaptationState:
        inverse_mass_matrix, wc_state = state
        wc_state = wc_update(wc_state, value)
        return MassMatrixAdaptationState(inverse_mass_matrix, wc_state)

    @partial(jax.jit, static_argnums=(0,))
    def final(state: MassMatrixAdaptationState) -> np.DeviceArray:
        inverse_mass_matrix, wc_state = state
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

        return inverse_mass_matrix

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
            m2 = m2 + delta * updated_delta
        else:
            m2 = m2 + np.outer(delta, updated_delta)

        return WelfordAlgorithmState(mean, m2, count)

    def covariance(state: WelfordAlgorithmState) -> np.DeviceArray:
        mean, m2, count = state
        covariance = m2 / (count - 1)
        return covariance, count, mean

    return init, update, covariance


# ------------------------------------------
#      == INTEGRATION STEPS ADAPTATION ==
# ------------------------------------------


def longest_batch_before_turn(integrator_step: Callable) -> Callable:
    """Learn the number of steps one can make before the trajectory makes a
    U-Turn. This routine is part of the adaptive strategy described in [1]_:
    during the warmup phase we run this scheme many times in order to get a
    distribution of numbers of steps before U-Turn. We then sample elements
    from this distribution during inference to use as the number of integration
    steps.

    References
    ----------
    .. [1]: Wu, Changye, Julien Stoehr, and Christian P. Robert. "Faster
            Hamiltonian Monte Carlo by learning leapfrog scale." arXiv preprint
            arXiv:1810.04449 (2018).
    """

    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def run(
        initial_position: np.DeviceArray,
        initial_momentum: np.DeviceArray,
        step_size: float,
        num_integration_steps: int,
    ):
        def cond(state: Tuple) -> bool:
            iteration, position, momentum = state
            return is_u_turn or iteration == num_integration_steps

        def update(state: Tuple) -> Tuple:
            iteration, position, momentum = state
            iteration += 1
            position, momentum = integrator_step(position, momentum, step_size, 1)
            return (iteration, position, momentum)

        result = jax.lax.while_loop(
            cond, update, (0, initial_position, initial_momentum)
        )

        return result[0]

    return run


@partial(jax.jit, static_argnums=(0, 2))
def is_u_turn(
    initial_position: np.DeviceArray,
    position: np.DeviceArray,
    inverse_mass_matrix: np.DeviceArray,
    momentum: np.DeviceArray,
) -> bool:
    """Detect when the trajectory starts turning back towards the point
    where it started.
    """
    v = np.multiply(inverse_mass_matrix, momentum)
    position_vec = position - initial_position
    projection = np.multiply(position_vec, v)
    return np.where(projection < 0, True, False)
