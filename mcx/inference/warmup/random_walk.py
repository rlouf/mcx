"""Warmup for the random walk Metropolis Hatings sampling algorithm."""
from typing import NamedTuple

import jax
import jax.numpy as np

from mcx.inference.optimizers import Adam, AdamState


class RWMHWarmupState(NamedTuple):
    parameter: np.DeviceArray
    adam_state: AdamState


def rwmh_warmup(kernel_factory, target_acceptance_rate):
    """How do we make it such that the different values are
    updated independently? All depends on the gradient. Needs
    to be computed per variable?
    """

    def init():
        pass

    @jax.jit
    def update():
        pass

    adam_init, adam_update = Adam(0.01)
    adam_state = adam_init(sigma)

    kernel = kernel_factory(sigma)
    state, info = kernel(state)
    is_accepted = info.is_accepted
    acceptance_rate = None  # Maybe an exponential average

    # do this 10-100 time
    gradient = target_acceptance_rate - acceptance_rate
    sigma, adam_state = adam_update(adam_state, sigma, gradient)
