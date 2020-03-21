import ast
import functools
from types import FunctionType
from typing import Dict, List, Union

import jax
import numpy

from mcx import core
from mcx.distributions import Distribution


__all__ = ["model", "sample_forward", "seed"]


class model(Distribution):
    """Representation of a model.

    Since it represents a probability graphical model, the `model` instance is
    a (multivariate) probability distribution, and as such inherits from the
    `Distribution` class. It implements the `sample` and `logpdf` methods.

    Models are expressed as functions. The expression of the model within the
    function should be as close to the mathematical expression as possible. The
    only difference with standard python code is the use of the "@" operator
    for random variable assignments.

    The models are then parsed into an internal graph representation that can
    be conditioned on data, compiled into a logpdf or a forward sampler. The
    result is pure functions that can be further JIT-compiled with JAX,
    differentiated and dispatched on GPUs and TPUs.

    The graph can be inspected and modified at runtime.

    Arguments
    ---------
    model: A function that contains `mcx` model definition.

    Examples
    --------

    Let us define a linear model in 1 dimension. In `mcx`, models are expressed
    in their generative form, that is a function that transforms some
    (optional) input---data, parameters---and returns the result:

    >>> import jax.numpy as np
    >>> import mcx
    >>> import mcx.distributions as dist
    >>>
    >>> def linear_model(X):
    ...     weights @ dist.Normal(0, 1)
    ...     sigma @ dist.Exponential(1)
    ...     z = np.dot(X, weights)
    ...     y @ Normal(z, sigma)
    ...     return y

    The symbol `@` is used here does not stand for the matrix multiplication but for
    the assignment of a random variable. The model can then be instantiated by calling:

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
    ...     weights @ dist.Normal(0, 1)
    ...     sigma @ dist.Exponential(1)
    ...     z = np.dot(X, weights)
    ...     y @ Normal(z, sigma)
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

    >>> mcx.sample_forward(linear_model, x=np.array([1, 2, 3]), num_samples=1000)

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
    ...     weights @ Normal(0, 1)
    ...     sigma @ HalfNormal(0, 1)
    ...     z = mult(X, weights)
    ...     y @ Normal(z, sigma)
    ...     return y

    Models also implicitly define a multivariate distribution. Following
    PyMC4's philosophy [2]_, we can use other models as distributions when
    defining a random variable. More precisely, what is meant by `x @ linear_model(1)`
    is "x is distributed according to :math:`P_{linear_model}(y|weights, sigma, x=1)`:

    >>> from my_module import hyperprior
    ...
    ... @mcx.model
    ... def prior(a):
    ...     s @ hyperprior()
    ...     p @ dist.Normal(a,a)
    ...     return p
    ...
    ... @mcx.model
    ... def linear_model(X):
    ...     weights @ prior(1)
    ...     sigma @ HalfNormal(0, 1)
    ...     z = mult(X, weights)
    ...     y @ Normal(z, sigma)
    ...     return y

    References
    ----------
    .. [1] van de Meent, Jan-Willem, Brooks Paige, Hongseok Yang, and Frank
           Wood. "An introduction to probabilistic programming." arXiv preprint
           arXiv:1809.10756 (2018).
    .. [2] Kochurov, Max, Colin Carroll, Thomas Wiecki, and Junpeng Lao.
           "PyMC4: Exploiting Coroutines for Implementing a Probabilistic Programming
           Framework." (2019).
    """

    def __init__(self, fn: FunctionType) -> None:
        self.model_fn = fn
        self.namespace = fn.__globals__
        self.graph = core.parse_definition(fn, self.namespace)
        self.rng_key = jax.random.PRNGKey(53)  # random.org
        functools.update_wrapper(self, fn)

    def __call__(self, *args) -> numpy.ndarray:
        """Return a sample from the prior predictive distribution. A different
        value is returned at each all.
        """
        _, self.rng_key = jax.random.split(self.rng_key)

        forward_sampler, _, _, _ = core.compile_to_forward_sampler(
            self.graph, self.namespace
        )
        samples = forward_sampler(self.rng_key, (1,), *args)
        return numpy.asarray(samples).squeeze()

    def __getitem__(self, name: str):
        """Access the graph by variable name.
        """
        nodes = self.graph.nodes(data=True)
        return nodes[name]["content"]

    def __setitem__(self, name: str, expression_str: str) -> None:
        """Change the distribution of one of the model's variables.

        The distribution and its arguments must be expressed within
        a string:

        >>> model['x'] = "dist.Normal(0, 1)"

        Note
        ----
        The parsing logic should be moved to the core.
        """
        expression_ast = ast.parse(expression_str).body[0]
        if isinstance(expression_ast, ast.Expr):
            expression = expression_ast.value
            if isinstance(expression, ast.Call):
                expression_arguments = expression.args

        self.graph.remove_node(name)
        distribution = expression
        arguments: List[Union[str, float, int, complex]] = []
        for arg in expression_arguments:
            if isinstance(arg, ast.Name):
                arguments.append(arg.id)
            elif isinstance(arg, ast.Constant):
                arguments.append(arg.value)
            elif isinstance(arg, ast.Num):
                arguments.append(arg.n)
        self.graph.add_randvar(name, distribution, arguments)

    def do(self, **kwargs) -> "model":
        """Apply the do operator to the graph and return a copy of the model.

        The do-operator :math:`do(X=x)` sets the value of the variable X to x and
        removes all edges between X and its parents [1]_. Applying the do-operator
        may be useful to analyze the behavior of the model before inference.

        References
        ----------
        .. [1]: Pearl, Judea. Causality. Cambridge university press, 2009.
        """
        conditionned_graph = self.graph.do(**kwargs)
        new_model = model(self.model_fn)
        new_model.graph = conditionned_graph
        return new_model

    def sample(self, *args, sample_shape=(1000,)) -> jax.numpy.DeviceArray:
        """Return forward samples from the distribution.
        """
        sampler, _, _, _ = core.compile_to_sampler(self.graph, self.namespace)
        _, self.rng_key = jax.random.split(self.rng_key)
        samples = sampler(self.rng_key, sample_shape, *args)
        return samples

    def logpdf(self, *args, **kwargs) -> float:
        """Compute the value of the distribution's logpdf.
        """
        logpdf, _, _, _ = core.compile_to_logpdf(self.graph, self.namespace)
        return logpdf(*args, **kwargs)

    @property
    def logpdf_src(self) -> str:
        """Return the source code of the log-probability density funtion
        generated by the compiler.
        """
        artifact = core.compile_to_logpdf(self.graph, self.namespace)
        return artifact.fn_source

    @property
    def sampler_src(self) -> str:
        """Return the source code of the forward sampling funtion
        generated by the compiler.
        """
        artifact = core.compile_to_sampler(self.graph, self.namespace)
        return artifact.fn_source

    @property
    def nodes(self):
        """Return the names of the nodes in the graph."""
        return self.graph.nodes

    @property
    def arguments(self):
        """Return the names of the graph's arguments."""
        return self.graph.arguments

    @property
    def posargs(self):
        """Return the names of the graph's positional arguments."""
        return self.graph.posargs

    @property
    def returned_variables(self):
        """Return the names of the graph's return variables."""
        return self.graph.returned_variables

    @property
    def variables(self):
        """Return the names of the random variables and transformed variables.
        """
        return self.graph.variables

    @property
    def random_variables(self):
        """Return the names of the random variables.
        """
        return self.graph.random_variables

    @property
    def posterior_variables(self):
        """Return the names of the random variables whose posterior we sample.
        """
        return self.graph.posterior_variables


# Convenience functions


def sample_forward(model: model, num_samples=1000, **kwargs) -> Dict:
    """Returns forward samples from the model.

    The samples are returned in a dictionary, with the names of
    the variables as keys.
    """
    args = list(kwargs.values())
    sample_shape = (num_samples,)
    samples = model.sample(*args, sample_shape=sample_shape)

    trace = {}
    for arg, arg_samples in zip(model.variables, samples):
        trace[arg] = numpy.asarray(arg_samples).T.squeeze()
    return trace


def seed(model: model, rng_key: jax.random.PRNGKey) -> model:
    """Set the random seed of the model."""
    model.rng_key = rng_key
    return model
