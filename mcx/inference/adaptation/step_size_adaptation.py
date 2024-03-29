"""Adaptive algorithms for the step size in the Hamiltonian Monte Carlo sampling algorithms family.

The Stan Manual [1]_ is a very good reference on automatic tuning of
parameters used in Hamiltonian Monte Carlo.

.. [1]: "HMC Algorithm Parameters", Stan Manual
        https://mc-stan.org/docs/2_20/reference-manual/hmc-algorithm-parameters.html
"""
from functools import partial
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from mcx.inference.kernels import HMCState

__all__ = [
    "dual_averaging",
    "find_reasonable_step_size",
]


# --------------------------------------
#      == DUAL AVERAGING ==
# --------------------------------------


class DualAveragingState(NamedTuple):
    """State carried through the dual averaging procedure.

    log_step_size
        The logarithm of the current value of the step size.
    log_step_size_avg
        The time-weighted average of the values that the logarithm of the step size
        has taken so far.
    step
        The current iteration step.
    avg_err
        The time average of the value of the quantity :math:`H_t`, the difference between
        the target acceptance rate and the current acceptance rate.
    mu
        Arbitrary point the values of log_step_size are shrunk towards. Chose to be
        :math:`\\log(10 \\epsilon_0)` where :math:`\\epsilon_0` is chosen in this context
        to be the step size given by the `find_reasonable_step_size` procedure.
    """

    log_step_size: float
    log_step_size_avg: float
    step: int
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

    Parameters
    ----------
    t0: float >= 0
        Free parameter that stabilizes the initial iterations of the algorithm.
        Large values may slow down convergence. Introduced in [2]_ with a default
        value of 10.
    gamma
        Controls the speed of convergence of the scheme. The authors of [2]_ recommend
        a value of 0.05.
    kappa: float in ]0.5, 1]
        Controls the weights of past steps in the current update. The scheme will
        quickly forget earlier step for a small value of `kappa`. Introduced
        in [2]_, with a recommended value of .75

    Returns
    -------
    init
        A function that initializes the state of the dual averaging scheme.
    update
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

        The parameter :math:`\\mu` is set to :math:`\\log(10 \\epsilon_1)`
        where :math:`\\epsilon_1` is the initial value of the step size.
        """
        mu: float = jnp.log(10 * inital_step_size)
        step = 1
        avg_error: float = 0.0
        log_step_size: float = jnp.log(inital_step_size)
        log_step_size_avg: float = 0.0
        return DualAveragingState(log_step_size, log_step_size_avg, step, avg_error, mu)

    @jax.jit
    def update(p_accept: float, state: DualAveragingState) -> DualAveragingState:
        """Update the state of the Dual Averaging adaptive algorithm.

        Parameters
        ----------
        p_accept: float in [0, 1]
            The current metropolis acceptance rate.
        state:
            The current state of the dual averaging algorithm.

        Returns
        -------
        The updated state of the dual averaging algorithm.
        """
        log_step, avg_log_step, step, avg_error, mu = state
        reg_step = step + t0
        eta_t = step ** (-kappa)
        avg_error = (1 - (1 / (reg_step))) * avg_error + (target - p_accept) / reg_step
        log_step_size = mu - (jnp.sqrt(step) / gamma) * avg_error
        log_step_size_avg = eta_t * log_step + (1 - eta_t) * avg_log_step
        return DualAveragingState(
            log_step_size, log_step_size_avg, step + 1, avg_error, mu
        )

    return init, update


# --------------------------------------------
#      == FIND REASONABLE INITIAL VALUE ==
# --------------------------------------------


class ReasonableStepSizeState(NamedTuple):
    """State carried through the search for a reasonable first step size.

    rng_key
        Key used by JAX's random number generator.
    direction: {-1, 1}
        Determines whether the step size should be increased or decreased during the
        previous step search. If direction = 1 it will be increased, otherwise decreased.
    previous_direction
        The previous direction. It is necessary to carry it because the choice of step
        size is made at the end of the search update.
    step_size
        The current step size in the search.
    """

    rng_key: jnp.ndarray
    direction: int
    previous_direction: int
    step_size: float


@partial(jax.jit, static_argnums=(1,))
def find_reasonable_step_size(
    rng_key: jnp.ndarray,
    kernel_generator: Callable[[float, jnp.DeviceArray], Callable],
    reference_hmc_state: HMCState,
    inverse_mass_matrix: jnp.DeviceArray,
    initial_step_size: float = 1.0,
    target_accept: float = 0.65,
) -> float:
    """Find a reasonable initial step size during warmup.

    While the dual averaging scheme is guaranteed to converge to a reasonable
    value for the step size starting from any value, choosing a good first
    value can speed up the convergence. This heuristics doubles and halves the
    step size until the acceptance probability of the HMC proposal crosses the
    .5 value.

    Parameters
    ----------
    rng_key
       Key used by JAX's random number generator.
    kernel_generator
        A function that takes a step size as an input and returns the corresponding
        sampling kernel.
    reference_hmc_state
        The location (HMC state) where this first step size must be found. This function
        never advances the chain.
    inverse_mass_matrix
        The inverse mass matrix relative to which the step size must be found.
    initial_step_size
        The first step size used to start the search.
    target_accept
        Once that value of the metropolis acceptance probability is reached we
        estimate that we have found a "reasonable" first step size.

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
    fp_limit = jnp.finfo(jax.lax.dtype(initial_step_size))

    @jax.jit
    def _update(search_state: ReasonableStepSizeState) -> ReasonableStepSizeState:
        """Perform one step of the step size search."""
        rng_key, direction, _, step_size = search_state
        _, rng_key = jax.random.split(rng_key)

        step_size = (2.0 ** direction) * step_size
        kernel = kernel_generator(step_size, inverse_mass_matrix)
        _, hmc_info = kernel(rng_key, reference_hmc_state)

        new_direction = jnp.where(
            target_accept < hmc_info.acceptance_probability, 1, -1
        )
        return ReasonableStepSizeState(rng_key, new_direction, direction, step_size)

    @jax.jit
    def _do_continue(search_state: ReasonableStepSizeState) -> bool:
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
        _do_continue, _update, ReasonableStepSizeState(rng_key, 0, 0, initial_step_size)
    )
    return step_size
