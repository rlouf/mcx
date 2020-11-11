"""Warmup for the random walk Metropolis Hatings sampling algorithm."""
from typing import NamedTuple

import jax
import jax.numpy as np

from mcx.inference.optimizers import Adam, AdamState


class RWMHWarmupState(NamedTuple):
    parameter: np.DeviceArray
    adam_state: AdamState
    acceptance_rate: float  # running acceptance rate
    step: int


def rwmh_warmup(kernel_factory, target_acceptance_rate, **adam_args):
    """Random Walk Metropolis Hastings warmup.

    We use the Adam optimizer pour minimize the difference between the running
    acceptance rate and the target acceptance rate. The running acceptance rate
    is computed as the average number of times moves were chosen during the
    acceptance step.

    Parameters
    ----------
    kernel_factory
        A function that returns a transition kernel given a value of the parameter.
    target_acceptance_rate
        The acceptance rate that we wish to attain by varying the value of the
        parameter.
    **adam_args
        Arguments used to instantiate the Adam optimizer.

    """
    adam_init, adam_update = Adam(**adam_args)

    def init(initial_parameter):
        adam_state = adam_init(initial_parameter)
        return RWMHWarmupState(initial_parameter, adam_state, 0.0, 0)

    @jax.jit
    def update(chain_state, warmup_state):
        parameter, adam_state, acceptance_rate, step = warmup_state
        kernel = kernel_factory(parameter)
        new_chain_state, info = kernel(chain_state)

        new_acceptance_rate = (step * acceptance_rate + info.is_accepted) / (step + 1)
        gradient = target_acceptance_rate - new_acceptance_rate
        new_parameter, new_adam_state = adam_update(adam_state, parameter, gradient)

        return new_chain_state, RWMHWarmupState(
            new_parameter, new_adam_state, new_acceptance_rate, step + 1
        )

    def final(warmup_state: RWMHWarmupState):
        return warmup_state.parameter
