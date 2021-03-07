"""Sample from the predictive distributions defined by the generative function."""
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp

import mcx
from mcx.sample import validate_model_args
from mcx.trace import Trace

__all__ = ["posterior_predict", "predict", "prior_predict"]


def prior_predict(
    rng_key: jnp.ndarray,
    model: mcx.model,
    model_args: Tuple[Any],
    num_samples: int = 1000,
):
    return predict(rng_key, model, model_args, num_samples)


def posterior_predict(
    rng_key: jnp.ndarray,
    model: mcx.model,
    model_args: Tuple[Any],
    trace: Trace,
    num_samples: int = 100,
):
    evaluated_model = mcx.evaluate(model, trace)
    return predict(rng_key, evaluated_model, model_args, num_samples)


def predict(
    rng_key: jnp.ndarray,
    model: Union[mcx.model, mcx.generative_function],
    model_args: Tuple[Any],
    num_samples: int = 100,
):
    """Provides a unified interface to sample from the prior and posterior
    predictive distributions.

    The rationale behind using the same API for both prior and posterior
    predictive sampling is simple: both are the results of simulations of the
    generative models. The prior predictive distribution corresponds to running
    the function by sampling from the prior distribution of each parameter; the
    posterior predictive distribution to simulations where each parameter is
    distributed according to its posterior distribution.

    The input data is broadcasted using numpy's broadcasting rules, and we
    then draw `num_samples` samples from the prior predictive distribution
    for each data point so the resulting array is of shape
    `(data_broadcasted_shape, num_samples)`.

    Parameters
    ----------
    rng_key
        The key used to seed JAX's random number generator.
    model
        The model that is used for predictions.
    model_args
        A tuple that contains the arguments passed to the model. It can either
        be input data or parameters.
    num_samples
        The number of samples to take from the predictive distribution.

    Returns
    -------
    An array of shape (num_samples, var_shape) from the predictive distribution.
    """
    _ = validate_model_args(model, model_args)
    keys = jax.random.split(rng_key, num_samples)

    in_axes: Tuple[int, ...] = (0,)
    sampler_args: Tuple[Any, ...] = (keys,)
    for arg in model_args:
        try:
            sampler_args += (jnp.atleast_1d(arg),)
        except RuntimeError:
            sampler_args += (arg,)
        in_axes += (None,)

    # TODO(remi): Handle the case where multiple values are returned.
    samples = jax.vmap(model, in_axes, out_axes=0)(*sampler_args)

    return samples
