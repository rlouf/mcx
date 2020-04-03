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
        state: AdamState, weights: np.DeviceArray, gradients: np.DeviceArray
    ) -> Tuple[np.DeviceArray, AdamState]:
        m, v, step = state

        m = (1 - b1) * gradients + b1
        v = (1 - b2) * (gradients ** 2) + b2

        mhat = m / (1 - b1 ** (step + 1))
        vhat = v / (1 - b2 ** (step + 1))
        new_weights = (1 - weight_decay_rate) * weights - (
            learning_rate * mhat / (np.sqrt(vhat) + eps)
        ).astype(weights.dtype)

        return new_weights, AdamState(m, v, step + 1)

    return init, update


class RMSPropState(NamedTuple):
    avg_squared_gradients: float


def RMSProp(learning_rate: float = 0.001, gamma: float = 0.9, eps: float = 1e-5):
    def init(weights: np.DeviceArray) -> RMSPropState:
        return RMSPropState(np.ones_like(weights))

    def update(
        state: RMSPropState, weights: np.DeviceArray, gradients: np.DeviceArray
    ) -> Tuple[np.DeviceArray, RMSPropState]:
        avg_squared_gradients = state[0]
        avg_squared_gradients = avg_squared_gradients * gamma + gradients ** 2 * (
            1.0 - gamma
        )
        new_weights = weights - (
            learning_rate * gradients / (np.sqrt(avg_squared_gradients) + eps)
        ).astype(weights.dtype)

        return new_weights, RMSPropState(avg_squared_gradients)

    return init, update
