from jax import nn
from jax import numpy as np
from jax import random

from . import constraints
from .distribution import Distribution


class Categorical(Distribution):
    param_constraints = {"logits": constraints.real}

    def __init__(self, logits):
        self.support = constraints.integer_interval(0, logits.shape[-1])

        self.event_shape = ()
        self.batch_shape = logits.shape[:-1]
        self.logits = logits

    def sample(self, rng_key, sample_shape):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.categorical(rng_key, self.logits, axis=0, shape=shape)

    def logpdf(self, x):
        log_p = nn.log_softmax(self.logits, axis=0)
        return np.sum(x * log_p)
