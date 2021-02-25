"""Sample from the predictive distributions defined by the generative function."""
import jax
import jax.numpy as jnp

import mcx
from mcx.trace import Trace

__all__ = ["posterior_predict", "prior_predict"]


def prior_predict(
    rng_key: jnp.ndarray,
    model: mcx.model,
    num_samples: int = 1000,
):
    keys = jax.random.split(rng_key, num_samples)
    return jax.vmap(model.sample)(keys)


def posterior_predict(
    rng_key: jnp.ndarray,
    model: mcx.model,
    trace: Trace,
    num_samples: int = 1000,
):
    keys = jax.random.split(rng_key, num_samples)
    sample_predictive = model.evaluate(trace)
    return jax.vmap(sample_predictive)(keys)
