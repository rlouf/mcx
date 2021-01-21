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
    sample_fn, _ = mcx.core.sample_joint(model)
    return sample_fn


def predictive_sampler(model: "model") -> FunctionType:
    call_fn, _ = mcx.core.sample(model)
    return call_fn


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

    Examples
    --------
    Let us define a linear model in 1 dimension. In `mcx`, models are expressed
    in their generative form, that is a function that transforms some
    (optional) input---data, parameters---and returns the result:

    >>> import jax.numpy as jnp
    >>> import mcx
    >>> import mcx.distributions as dist
    >>>
    >>> def linear_model(X):
    ...     weights <~ dist.Normal(0, 1)
    ...     sigma <~ dist.Exponential(1)
    ...     z = jnp.dot(X, weights)
    ...     y <~ Normal(z, sigma)
    ...     return y

    The symbol `<~` is used here does not stand for the combination of the `<`
    comparison and `~` invert operators but for the assignment of a random
    variable. The model can then be instantiated by calling:

    >>> model = mcx.model(linear_model)

    Generative models are stochastic functions, you can call them like you would any function:

    >>> model(1)
    -2.31

    If you call it again, it will give you a different result:

    >>> model(1)
    1.57

    We say that these results are drawn from the prior predictive distribution for x=1.
    More formally, :math:`P(y|weights, sigma, x=1)`. If you add the decorator `@mcx.model`
    on top of the function:

    >>> @mcx.model
    ... def linear_model(X):
    ...     weights <~ dist.Normal(0, 1)
    ...     sigma <~ dist.Exponential(1)
    ...     z = jnp.dot(X, weights)
    ...     y <~ Normal(z, sigma)
    ...     return y

    You can directly call the function:

    >>> linear_model(1)
    1.57

    While this recompiles the graph at each call, the performance hit is not
    noticeable in practice.

    Calling the function directly is useful for quick sanity check and debugging, but
    we often need a more complete view of the prior predictive distribution, or the
    forward sampling distribution of each parameter in the model:

    >>> mcx.sample_forward(linear_model, x=1, num_samples=1000)
    {'weight': array([1, ....]), 'sigma': array([2.1, ...]), 'y': array([1.56, ...])}

    This also works for an array input; standard broadcasting rules apply:

    >>> mcx.sample_forward(linear_model, x=jnp.array([1, 2, 3]), num_samples=1000)

    Unlike calling the model directly, this function JIT-compiles the forward
    sampler; if your machine has a GPU, it will automatically run on it. This
    should allow for quick iteration on the initial phase of the modeling
    process.

    To explore the model, we can also use the "do" operator to fix the value of
    a random variable. This returns a copy of the model where all connections
    with the parent nodes have been removed:

    >>> conditioned = linear_model.do(sigma=1000)
    ... conditioned(1)
    435.7

    'mcx' translates your model definition into a graph. This graph can be explored
    and modified at runtime. You can explore nodes:

    >>> print(linear_model["weight"])
    [should have distibution, plus info about distirbution]

    And modify them:

    >>> linear_model["weight"] = "dist.Normal(0, 4)"

    Behind the scenes, `mcx` inspects the definition's source code and
    translates it to a graph. Since `mcx` sticks closely to python's syntax (in
    fact, only adds one construct), most of what you would do inside a function
    in Python is also valid in model definitions. For instance, complex
    transformations can be defined in functions that are called within the
    model:

    >>> from utils import mult
    ...
    ... @mcx.model
    ... def linear_model(X):
    ...     weights <~ Normal(0, 1)
    ...     sigma <~ HalfNormal(0, 1)
    ...     z = mult(X, weights)
    ...     y <~ Normal(z, sigma)
    ...     return y

    Models also implicitly define a multivariate distribution. Following
    PyMC4's philosophy [2]_, we can use other models as distributions when
    defining a random variable. More precisely, what is meant by `x <~ linear_model(1)`
    is "x is distributed according to :math:`P_{linear_model}(y|weights, sigma, x=1)`:

    >>> from my_module import hyperprior
    ...
    ... @mcx.model
    ... def prior(a):
    ...     s <~ hyperprior()
    ...     p <~ dist.Normal(a,a)
    ...     return p
    ...
    ... @mcx.model
    ... def linear_model(X):
    ...     weights <~ prior(1)
    ...     sigma <~ HalfNormal(0, 1)
    ...     z = np.dot(X, weights)
    ...     y <~ Normal(z, sigma)
    ...     return y


    The `model` class gives a no-fuss access to MCX's models. It is not
    compulsory, for instance, to use JAX's key splitting mechanism to obtain
    many samples.

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
        self.call_fn, self.call_src = mcx.core.sample(self)

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
        return self.call_fn(rng_key, *args, **kwargs)

    def sample(self, rng_key, *args, **kwargs) -> Dict:
        """Sample from the joint distribution."""
        return self.sample_joint_fn(rng_key, *args, **kwargs)

    def logpdf(self, *rv_and_args, **kwargs) -> jnp.DeviceArray:
        """Value of the log-probability density function of the distribution.

        TODO: Figure out the right interface for the logpdf, and document it.
        """
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

        def seeded_call(self, *args, **kwargs):
            rng_key = next(keys)
            return old_call(rng_key, *args, **kwargs)

        def seeded_sample(self, *args, **kwargs):
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


def seed(model: "model", rng_key: jax.random.PRNGKey):
    """Wrap the model's calling function to do the rng splitting automatically."""
    seeded_model = mcx.model(model.model_fn)  # type: ignore
    seeded_model.seed(rng_key)
    return seeded_model


# -------------------------------------------------------------------
#                  == GENERATIVE FUNCTION ==
# -------------------------------------------------------------------


class generative_function(object):
    def __init__(self, model_fn: FunctionType, trace: Trace) -> None:
        """Create a generative function.

        We create a generative function, or stochastic program, by conditioning
        the values of a model's random variables. A typical application is to
        create a function that returns samples from the posterior predictive
        distribution.

        """
        self.graph, self.namespace = mcx.core.parse(model_fn)
        self.model_fn = model_fn
        self.conditioning = trace

        self.call_fn, self.call_src = mcx.core.sample_posterior_predictive(
            self, trace.keys()
        )
        self.trace = trace

    def __call__(self, rng_key, *args, **kwargs) -> jnp.DeviceArray:
        """Call the model as a generative function."""
        print(args, kwargs, self.trace)
        return self.call_fn(rng_key, *args, **kwargs, **self.trace)


def evaluate(model: "model", trace: Trace):
    """Evaluate the model at the posterior distribution."""
    evaluated_model = mcx.generative_function(model.model_fn, trace)  # type: ignore
    return evaluated_model
