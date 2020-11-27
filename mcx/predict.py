from typing import Any, Dict, Tuple, Union

import jax
import numpy
from jax import numpy as np

import mcx.compiler as compiler
from mcx.trace import Trace

__all__ = ["predict", "sample_forward"]


# -------------------------------------------------------------------
#                == PRIOR & POSTERIOR PREDICTIONS ==
# -------------------------------------------------------------------


def predict(
    rng_key: jax.random.PRNGKey, model, trace: Trace = None
) -> Union["prior_predictive", "posterior_predictive"]:
    """Provides a unified interface for prior and posterior predictive sampling.

    The rationale behind using the same API for both prior and posterior
    predictive sampling is simple: both are prediction of the outputs of the
    generative models. Before and after the model has been evaluated on data.

    Note
    ----
    The clunkiness of any prior-posterior predictive sampling API comes from
    the 'static model + trace' abstraction and would disappear within an
    'evaluated program' paradigm.

    Parameters
    ----------
    rng_key
        The key used to seed JAX's random number generator.
    model
        The model that is used for predictions.
    trace
        Samples from the model's posterior distribution computed with the data
        used for predictions or with another dataset. If specified `predict` will
        return posterior predictive samples, otherwise prior predictive samples.

    Returns
    -------
    Either the prior predictive or posterior predictive class.

    """
    if isinstance(trace, Trace):
        return posterior_predictive(rng_key, model, trace)
    else:
        return prior_predictive(rng_key, model)


class posterior_predictive:
    def __init__(self, rng_key: jax.random.PRNGKey, model, trace: Trace) -> None:
        """Initialize the posterior predictive sampler."""
        artifact = compiler.compile_to_posterior_sampler(model.graph, model.namespace)
        sampler = jax.jit(artifact.compiled_fn)

        self.model = model
        self.rng_key = rng_key
        self.sampler = sampler
        self.trace = trace

    def __call__(self, **observations) -> Dict:
        """Generate posterior predictive samples from observations.

        Parameters
        ----------
        **observations
            The values of the model's input parameters for which we want to compute
            predictions.

        Returns
        -------
        A dictionary that maps each returned variable to its predicted values.

        """
        model_returnedvars = self.model.returned_variables

        sampler_args: Tuple[Any, ...] = (self.rng_key,)
        in_axes: Tuple[Any, ...] = (None,)
        in_axes_single_chain: Tuple[Any, ...] = (None,)

        # The sampling funtion generated by the compiler takes the following arguments
        # in order:
        #
        # 1. rng_key
        # 2. All of the model's definitions' positional argument
        # 3. All of the model's random variables
        # 4. All of the model's kwargs
        #
        # Here we pass all these arguments as positional arguments when using vmap. We
        # thus need to pass them in the correct order; for this we use the lists of
        # positional arguments, kwargs and random variables stored in the model.

        # Let us first handle the model's positional arguments
        model_posargs = self.model.posargs
        for arg in model_posargs:
            try:
                value = observations[arg]
                try:
                    sampler_args += (np.atleast_1d(value),)
                except RuntimeError:
                    sampler_args += (value,)
                in_axes_single_chain += (None,)
                in_axes += (None,)
            except KeyError:
                raise AttributeError(
                    "You need to specify a value for the variable {}".format(arg)
                )

        # Let us now add the random variables and their respective values found
        # in the trace.
        model_randvars = self.model.posterior_variables
        posterior_samples = self.trace.raw.samples
        for arg in model_randvars:
            try:
                value = posterior_samples[arg]  # type: ignore
                num_samples = np.shape(value)[-1]
                num_chains = np.shape(value)[0]
                try:
                    sampler_args += (np.atleast_1d(value),)
                    in_axes_single_chain += (-1,)  # mapping over samples
                    in_axes += (0,)  # mapping over chains
                except RuntimeError:
                    sampler_args += (value,)
                    in_axes_single_chain += (None,)
                    in_axes += (None,)
            except KeyError:
                raise AttributeError(
                    "You to provide posterior samples for the variable {}".format(arg)
                )

        # Let us now add the kwargs. If their value is not provided in
        # **observations we retrieve their default value in the graph.
        model_kwargs = tuple(set(self.model.arguments).difference(self.model.posargs))
        for kwarg in model_kwargs:
            if kwarg in observations:
                value = observations[kwarg]
            else:
                value = self.model.nodes[kwarg]["content"].default_value.n
            sampler_args += (np.atleast_1d(value),)
            in_axes_single_chain += (None,)
            in_axes += (None,)

        # out_axes specifies where the mapped axis should appear in the output.
        # This basically avoids us havig to transpose the arrays before returning
        # them to the user.
        out_axes_single_chain: Union[int, Tuple[int, ...]]
        if len(model_returnedvars) == 1:
            out_axes_single_chain = 1
        else:
            out_axes_single_chain = (1,) * len(model_returnedvars)

        def sample_one_chain(*args):
            return jax.vmap(
                self.sampler,
                in_axes=in_axes_single_chain,
                out_axes=out_axes_single_chain,
            )(*args)

        print(
            f"Generating {num_samples:,} predictive samples for the {num_chains:,} chains."
        )
        samples = jax.vmap(sample_one_chain, in_axes)(*sampler_args)

        predictive_trace = {
            arg: numpy.asarray(arg_samples).squeeze()
            for arg, arg_samples in zip(model_returnedvars, samples)
        }

        return predictive_trace


class prior_predictive:
    def __init__(self, rng_key: jax.random.PRNGKey, model) -> None:
        artifact = compiler.compile_to_prior_sampler(model.graph, model.namespace)
        sampler = jax.jit(artifact.compiled_fn)

        self.model = model
        self.rng_key = rng_key
        self.sampler = sampler

    def __call__(self, num_samples: int = 10, **observations) -> Dict:
        """Generate samples from the prior predictive distribution.

        The input data is broadcasted using numpy's broadcasting rules, and we
        then draw `num_samples` samples from the prior predictive distribution
        for each data point so the resulting array is of shape
        `(data_broadcasted_shape, num_samples)`.

        Parameters
        ----------
        num_samples
            The number of prior samples to draw from the posterior predictive
            distribution.
        **observations
            The values of the model's input parameters for which we want to compute
            predictions.

        Returns
        -------
        A dictionary that maps each returned variable to an array of shape
        (data_shape, num_samples) that contains samples from the models' prior
        predictive distribution.

        """
        model_returnedvars = self.model.returned_variables

        keys = jax.random.split(self.rng_key, num_samples)
        sampler_args: Tuple[Any, ...] = (keys,)
        in_axes: Tuple[int, ...] = (0,)

        # The sampling funtion generated by the compiler takes the following arguments
        # in order:
        #
        # 1. rng_key
        # 2. All of the model's definitions' positional argument
        # 4. All of the model's kwargs
        #
        # Here we pass all these arguments as positional arguments when using vmap. We
        # thus need to pass them in the correct order; for this we use the lists of
        # positional arguments and kwargs stored in the model.
        #
        # We map over the array of rng_keys to get as many samples from the
        # prior distribution.

        model_posargs = self.model.posargs
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

        model_kwargs = tuple(set(self.model.arguments).difference(self.model.posargs))
        for kwarg in model_kwargs:
            if kwarg in observations:
                value = observations[kwarg]
            else:
                # if the kwargs' values are not specified we retrieve them from the graph.
                value = self.model.nodes[kwarg]["content"].default_value.n
            sampler_args += (np.atleast_1d(value),)
            in_axes += (None,)

        print(f"Generating {num_samples:,} samples from the prior distribution.")
        samples = jax.vmap(self.sampler, in_axes, out_axes=1)(*sampler_args)

        predictive_trace = {
            arg: numpy.asarray(arg_samples).squeeze()
            for arg, arg_samples in zip(model_returnedvars, samples)
        }

        return predictive_trace


# -------------------------------------------------------------------
#                       == FORWARD SAMPLING ==
# -------------------------------------------------------------------


def sample_forward(
    rng_key: jax.random.PRNGKey, model, num_samples: int = 1, **observations
) -> Dict:
    """Returns forward samples from the model.

    Parameters
    ----------
    rng_key
        Key used by JAX's random number generator.
    model
        The model from which we want to get forward samples.
    num_samples
        The number of forward samples we want to draw for each variable.
    **observations
        The values of the model's input parameters.


    Returns
    -------
    A dictionary that maps all the (deterministic and random) variables defined
    in the model to samples from the forward prior distribution.

    """

    keys = jax.random.split(rng_key, num_samples)
    sampler_args: Tuple[Any, ...] = (keys,)
    in_axes: Tuple[int, ...] = (0,)

    model_posargs = model.posargs
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

    model_kwargs = tuple(set(model.arguments).difference(model.posargs))
    for kwarg in model_kwargs:
        if kwarg in observations:
            value = observations[kwarg]
        else:
            # if the kwarg value is not provided retrieve it from the graph.
            value = model.nodes[kwarg]["content"].default_value.n
        sampler_args += (value,)
        in_axes += (None,)

    out_axes: Union[int, Tuple[int, ...]]
    if len(model.variables) == 1:
        out_axes = 1
    else:
        out_axes = (1,) * len(model.variables)

    artifact = compiler.compile_to_sampler(model.graph, model.namespace)
    sampler = jax.jit(artifact.compiled_fn)
    samples = jax.vmap(sampler, in_axes, out_axes)(*sampler_args)

    forward_trace = {
        arg: numpy.asarray(arg_samples).squeeze()
        for arg, arg_samples in zip(model.variables, samples)
    }

    return forward_trace