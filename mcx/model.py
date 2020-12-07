import functools
from typing import Callable, Dict, Tuple
from types import FunctionType, MethodType

import jax
import jax.numpy as jnp
import mcx
from mcx.distributions import Distribution

__all__ = ["model"]


class model(Distribution):
    """Representation of a model.

    Since it represents a probability graphical model, the `model` instance is
    a (multivariate) probability distribution, and as such inherits from the
    `Distribution` class. It implements the `sample` and `logpdf` methods.

    Models are expressed as generative functions. The expression of the model
    within the function should be as close to the mathematical expression as
    possible. The only difference with standard python code is the use of the
    "<~" operator for random variable assignments.

    The models are then parsed into an internal graph representation that can
    be conditioned on data, compiled into a logpdf or a forward sampler. The
    result is pure functions that can be further JIT-compiled with JAX,
    differentiated and dispatched on GPUs and TPUs.

    The graph can be inspected and modified at runtime.

    Parameters
    ----------
    model: A function that contains `mcx` model definition.

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
        self.ir = mcx.core.parse(model_fn)
        self.model_fn = model_fn

        self.logpdf_fn, self.logpdf_src = mcx.core.logpdf(self.ir)
        self.sample_fn, self.sample_src = mcx.core.sample_forward(self.ir)
        self.call_fn, self.src = mcx.core.sample(self.ir)

        self.sample_fn = jax.jit(self.sample_fn)
        self.call_fn = jax.jit(self.call_fn)

        functools.update_wrapper(self, model_fn)

    def __call__(self, rng_key, *args, **kwargs) -> jnp.DeviceArray:
        """Call the model as a generative function."""
        return self.call(rng_key, *args, **kwargs)

    def call(self, rng_key, *args, **kwargs) -> jnp.DeviceArray:
        """We redirect the call to __call__ to allows monkey-patching
        when seeding the function. Indeed, special methods are attached
        to the class and not on particular instances so it is impossible
        to monkey patch them.
        """
        return self.call_fn(rng_key, *args, **kwargs)

    def logpdf(self, *rv_and_args, **kwargs) -> float:
        """Logpdf of the multivariate distribution defined by the model."""
        return self.logpdf_fn(*rv_and_args, **kwargs)

    def sample(self, rng_key, *args, **kwargs) -> Dict:
        """Sample from the predictive distribution."""
        return self.sample_fn(rng_key, *args, **kwargs)

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

    def do(self, **kwargs) -> "model":
        pass

    @property
    def args(self) -> Tuple[str]:
        return self.ir.graph.names['args']

    @property
    def kwargs(self) -> Tuple[str]:
        return self.ir.graph.names['kwargs']

    @property
    def random_variables(self) -> Tuple[str]:
        return self.ir.graph.names['random_variables']


def seed(model: "model", rng_key: jax.random.PRNGKey):
    """Wrap the model's calling function to do the rng splitting automatically.
    """
    seeded_model = mcx.model(model.model_fn)
    seeded_model.seed(rng_key)
    print("AH")
    return seeded_model


def evaluate(model: "model", trace: mcx.Trace):
    pass
