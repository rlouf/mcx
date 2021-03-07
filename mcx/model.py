import functools
from types import FunctionType, MethodType
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

import mcx
from mcx.distributions import Distribution
from mcx.trace import Trace

__all__ = [
    "model",
    "generative_function",
    "log_prob",
    "log_prob_contributions",
    "joint_sampler",
    "predictive_sampler",
]


# --------------------------------------------------------------------
#                        == TARGET FUNCTIONS ==
#
# The functions into which a MCX model can be compiled to sample from
# the different distributions it defines.
# --------------------------------------------------------------------


def log_prob(model: "model") -> FunctionType:
    logpdf_fn, _ = mcx.core.logpdf(model)
    return logpdf_fn


def log_prob_contributions(model: "model") -> FunctionType:
    logpdf_fn, _ = mcx.core.logpdf_contributions(model)
    return logpdf_fn


def joint_sampler(model: "model") -> FunctionType:
    sample_joint_fn, _ = mcx.core.sample_joint(model)
    return sample_joint_fn


def predictive_sampler(model: "model") -> FunctionType:
    sample_predictive_fn, _ = mcx.core.sample_predictive(model)
    return sample_predictive_fn


# -------------------------------------------------------------------
#                             == MODEL ==
# -------------------------------------------------------------------


class model(Distribution):
    """MCX representation of a probabilistic model (or program).

    Models are expressed as generative functions. The expression of the model
    within the function should be as close to the mathematical expression as
    possible. The only difference with standard python code is the use of the
    "<~" operator for random variable assignments. Model definitions are python
    functions decorated with the `@mcx.model` decorator. Calling the model returns
    samples from the prior predictive distribution.

    A model is a representation of a probabilistic graphical model, and as such
    implicitly defines a multivariate probability distribution. The class
    `model` thus inherits from the `Distribution` class and implements the
    `logpdf` and `sample` method. The `sample` method returns samples from the
    joint probability distribution.

    Since it represents a probability graphical model, the `model` instance is
    a (multivariate) probability distribution, and as such inherits from the
    `Distribution` class. It implements the `sample` and `logpdf` methods.

    Model expressions are parsed into an internal graph representation that can
    be conditioned on data, compiled into a logpdf or a forward sampler. The
    result is pure functions that can be further JIT-compiled with JAX,
    differentiated and dispatched on GPUs and TPUs. The graph can be inspected
    and modified at runtime.

    Attributes
    ----------
    model_fn:
        The function that contains `mcx` model definition.
    graph:
        The internal representation of the model as a graphical model.
    namespace:
        The namespace in which the function is called.

    Methods
    -------
    __call__:
        Return a sample from the prior predictive distribution.
    sample:
        Return a sampler from the joint probability distribution.
    logpdf:
        Return the value of the log-probability density function of the
        implied multivariate probability distribution.
    seed:
        Seed the model with an auto-updating PRNGKey so the sampling methods do
        not need to be called with a new key each time.

    References
    ----------
    .. [1] van de Meent, Jan-Willem, Brooks Paige, Hongseok Yang, and Frank
           Wood. "An introduction to probabilistic programming." arXiv preprint
           arXiv:1809.10756 (2018).
    .. [2] Kochurov, Max, Colin Carroll, Thomas Wiecki, and Junpeng Lao.
           "PyMC4: Exploiting Coroutines for Implementing a Probabilistic Programming
           Framework." (2019).
    """

    def __init__(self, model_fn: FunctionType) -> None:
        self.model_fn = model_fn
        self.graph, self.namespace = mcx.core.parse(model_fn)

        self.logpdf_fn, self.logpdf_src = mcx.core.logpdf(self)
        self.sample_joint_fn, self.sample_joint_src = mcx.core.sample_joint(self)
        (
            self.sample_predictive_fn,
            self.sample_predictive_src,
        ) = mcx.core.sample_predictive(self)

        functools.update_wrapper(self, model_fn)

    def __call__(self, rng_key, *args, **kwargs) -> jnp.DeviceArray:
        """Call the model as a generative function."""
        return self.call(rng_key, *args, **kwargs)

    def call(self, rng_key, *args, **kwargs) -> jnp.DeviceArray:
        """Returns a sample from the prior predictive distribution.

        We redirect the call to __call__ to be able to seed the function
        with a PRNG key. Indeed, special methods in python are attached
        to the class and not on particular instances making it impossible
        to monkey patch them.

        """
        return self.sample_predictive_fn(rng_key, *args, **kwargs)

    def sample(self, rng_key, *args, **kwargs) -> Dict:
        """Sample from the joint distribution."""
        return self.sample_joint_fn(rng_key, *args, **kwargs)

    def logpdf(self, *rv_and_args, **kwargs) -> jnp.DeviceArray:
        """Value of the log-probability density function of the distribution."""
        return self.logpdf_fn(*rv_and_args, **kwargs)

    def seed(self, rng_key):
        """Seed the model with a PRNGKey.

        It can be cumbersome to have to split the rng each time the
        sampling methods are called when doing exploratory analysis.
        We thus provide a convenience method that allows to seed
        these methods with keys that are split at every call.

        """

        def key_splitter(rng_key):
            while True:
                _, rng_key = jax.random.split(rng_key)
                yield rng_key

        keys = key_splitter(rng_key)
        old_sample = self.sample
        old_call = self.call

        def seeded_call(_, *args, **kwargs):
            rng_key = next(keys)
            return old_call(rng_key, *args, **kwargs)

        def seeded_sample(_, *args, **kwargs):
            rng_key = next(keys)
            return old_sample(rng_key, *args, **kwargs)

        self.call = MethodType(seeded_call, self)
        self.sample = MethodType(seeded_sample, self)

    @property
    def args(self) -> Tuple[str]:
        return self.graph.names["args"]

    @property
    def kwargs(self) -> Tuple[str]:
        return self.graph.names["kwargs"]

    @property
    def random_variables(self) -> Tuple[str]:
        return self.graph.names["random_variables"]


def seed(model: "model", rng_key: jnp.ndarray):
    """Wrap the model's calling function to do the rng splitting automatically."""
    seeded_model = mcx.model(model.model_fn)  # type: ignore
    seeded_model.seed(rng_key)
    return seeded_model


# -------------------------------------------------------------------
#                  == GENERATIVE FUNCTION ==
# -------------------------------------------------------------------


class generative_function(object):
    def __init__(self, model_fn: FunctionType, trace: Trace, chain_id=0) -> None:
        """Create a generative function.

        We create a generative function, or stochastic program, by conditioning
        the values of a model's random variables. A typical application is to
        create a function that returns samples from the posterior predictive
        distribution.

        """
        self.graph, self.namespace = mcx.core.parse(model_fn)
        self.model_fn = model_fn
        self.conditioning = trace

        self.call_fn, self.src = mcx.core.sample_posterior_predictive(
            self, trace.keys()
        )
        self.trace = trace
        self.chain_id = chain_id

    def __call__(self, rng_key, *args, **kwargs) -> jnp.DeviceArray:
        """Call the model as a generative function."""
        return self.call_fn(
            rng_key, *self.trace.chain_values(self.chain_id), *args, **kwargs
        )

    @property
    def args(self) -> Tuple[str]:
        return self.graph.names["args"]

    @property
    def kwargs(self) -> Tuple[str]:
        return self.graph.names["kwargs"]


def evaluate(model: "model", trace: Trace):
    """Evaluate the model at the posterior distribution."""
    evaluated_model = mcx.generative_function(model.model_fn, trace)  # type: ignore
    return evaluated_model
