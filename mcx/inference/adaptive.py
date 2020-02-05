"""Adaptive algorithms for Markov Chain Monte Carlo.
"""
from typing import Callable, Tuple

from jax import numpy as np

DAState = Tuple[float, float, int, float, float]


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

    def init(inital_step_size: float) -> DAState:
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
        return log_step_size, log_step_size_avg, t, avg_error, mu

    def update(p_accept: float, state: DAState) -> DAState:
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
        return log_step_size, log_step_size_avg, t, avg_error, mu

    return init, update
