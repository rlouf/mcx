"""Optimizers
"""
from typing import Callable, NamedTuple, Tuple

import jax.numpy as np


class AdamState(NamedTuple):
    m: float
    v: float
    step: int


def Adam(
    learning_rate: float,
    weight_decay_rate: float = 1e-5,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-5,
) -> Tuple[Callable, Callable]:
    def init(weights: np.DeviceArray) -> AdamState:
        m = np.zeros_like(weights)
        v = np.zeros_like(weights)
        return AdamState(m, v, 0)

    def update(
        state: AdamState, weights: np.DeviceArray, grads: np.DeviceArray
    ) -> Tuple[np.DeviceArray, AdamState]:
        m, v, step = state

        m = (1 - b1) * grads + b1
        v = (1 - b2) * (grads ** 2) + b2

        mhat = m / (1 - b1 ** (step + 1))
        vhat = v / (1 - b2 ** (step + 1))
        new_weights = (1 - weight_decay_rate) * weights - (
            learning_rate * mhat / (np.sqrt(vhat) + eps)
        ).astype(weights.dtype)

        return new_weights, AdamState(m, v, step + 1)

    return init, update
