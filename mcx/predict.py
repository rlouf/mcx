from typing import Any, Dict, Tuple, Union

import jax
from jax import numpy as np
import numpy

import mcx.core as core
from mcx.model import model as Model
from mcx.trace import Trace

__all__ = ["predict"]


def predict(
    rng_key: jax.random.PRNGKey, model: Model, trace: Trace = None
) -> Union["prior_predictive", "posterior_predictive"]:
    """Provide a unified interface for prior and posterior predictive sampling."""
    if isinstance(trace, Trace):
        return posterior_predictive(rng_key, model, trace)
    else:
        return prior_predictive(rng_key, model)


class posterior_predictive:
    def __init__(self, rng_key: jax.random.PRNGKey, model: Model, trace: Trace) -> None:
        artifact = core.compile_to_posterior_sampler(model.graph, model.namespace)
        sampler_fn = jax.jit(artifact.compiled_fn)

        self.model = model
        self.rng_key = rng_key
        self.sampler_fn = sampler_fn
        self.trace = trace

    def __call__(self, **observations) -> Dict:
        model_posargs = self.model.posargs
        model_randvars = self.model.posterior_variables
        model_returnedvars = self.model.returned_variables
        model_kwargs = tuple(set(self.model.arguments).difference(self.model.posargs))

        sampler_args: Tuple[Any, ...] = (self.rng_key,)
        in_axes: Tuple[Any, ...] = (None,)
        in_axes_chain: Tuple[Any, ...] = (None,)

        posterior_samples = self.trace.raw.samples

        for arg in model_posargs:
            try:
                value = observations[arg]
                try:
                    sampler_args += (np.atleast_1d(value),)
                except RuntimeError:
                    sampler_args += (value,)
                in_axes += (None,)
                in_axes_chain += (None,)
            except KeyError:
                raise AttributeError(
                    "You need to specify the value of the variable {}".format(arg)
                )

        for arg in model_randvars:
            try:
                value = posterior_samples[arg]  # type: ignore
                num_samples = np.shape(value)[-1]
                num_chains = np.shape(value)[0]
                try:
                    sampler_args += (np.atleast_1d(value),)
                except RuntimeError:
                    sampler_args += (value,)  # We need to vmap over chains
                in_axes += (-1,)
                in_axes_chain += (0,)
            except KeyError:
                raise AttributeError(
                    "You to provide posterior samples for the variable {}".format(arg)
                )

        for kwarg in model_kwargs:
            if kwarg in observations:
                value = observations[kwarg]
            else:
                value = self.model.nodes[kwarg]["content"].default_value.n
            sampler_args += (np.atleast_1d(value),)
            in_axes += (None,)
            in_axes_chain += (None,)

        if len(model_returnedvars) == 1:
            out_axes: int = 1
        else:
            out_axes: Tuple[int, ...] = (1,) * len(model_returnedvars)  # type: ignore

        def samples_one_chain(*args):
            # out_axes is brittle, it is going to fail if more than 1 returned variable
            return jax.vmap(self.sampler_fn, in_axes=in_axes, out_axes=out_axes)(
                *args
            ).squeeze()

        print(
            f"Generating {num_samples:,} predictive samples for the {num_chains:,} chains."
        )
        samples = jax.vmap(samples_one_chain, in_axes=in_axes_chain)(*sampler_args)

        predictive_trace = {
            arg: numpy.asarray(arg_samples).T.squeeze()
            for arg, arg_samples in zip(model_returnedvars, samples)
        }

        return predictive_trace


class prior_predictive:
    def __init__(self, rng_key: jax.random.PRNGKey, model: Model) -> None:
        artifact = core.compile_to_prior_sampler(model.graph, model.namespace)
        sampler_fn = artifact.compiled_fn
        sampler_fn = jax.jit(sampler_fn)

        self.model = model
        self.sampler_fn = sampler_fn
        self.rng_key = rng_key

    def __call__(self, num_samples: int = 10, **observations) -> Dict:
        """Generate samples from the prior predictive distribution.

        Returns
        -------
        samples
            A DeviceArray of shape (data_shape, num_samples) that contains samples from
            the models' prior predictive distribution.
        """
        model_posargs = self.model.posargs
        model_kwargs = tuple(set(self.model.arguments).difference(self.model.posargs))
        model_returnedvars = self.model.returned_variables

        keys = jax.random.split(self.rng_key, num_samples)
        sampler_args: Tuple[Any, ...] = (keys,)
        in_axes: Tuple[int, ...] = (0,)

        for arg in model_posargs:
            try:
                value = observations[arg]
                try:
                    sampler_args += (np.atleast_1d(value),)
                except RuntimeError:
                    sampler_args += (value,)
                in_axes += (None,)
            except KeyError:
                raise AttributeError(
                    "You need to specify the value of the variable {}".format(arg)
                )

        for kwarg in model_kwargs:
            if kwarg in observations:
                value = observations[kwarg]
            else:
                value = self.model.nodes[kwarg]["content"].default_value.n
            sampler_args += (np.atleast_1d(value),)
            in_axes += (None,)

        print(f"Generating {num_samples:,} samples from the prior distribution.")
        samples = jax.vmap(self.sampler_fn, in_axes=in_axes, out_axes=1)(*sampler_args)

        predictive_trace = {
            arg: numpy.asarray(arg_samples)
            for arg, arg_samples in zip(model_returnedvars, samples)
        }

        return predictive_trace
