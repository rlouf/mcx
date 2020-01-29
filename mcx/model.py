import functools
from typing import Callable
from types import FunctionType

import numpy

from mcx.compiler import to_graph, to_logpdf, to_prior_sampler
from mcx.distributions import Distribution


class model(Distribution):
    """Representation of a model.

    Since it represents a probability graphical model, the `model` instance is
    a (multivariate) probability distribution, and as such inherits from the
    `Distribution` class. It implements the `sample` and `logpdf` methods.

    Models are expressed as functions. The expression of the model within the
    function should be as close to the mathematical expression as possible. The
    only difference with standard python code is the use of the "@" operator
    for random variable assignments.

    The models are then parsed into an internal representation that can be
    conditioned on data, compiled into a logpdf or a forward sampler. The
    result is pure functions that can be further JIT-compiled with JAX,
    differentiated and dispatched on GPUs and TPUs.

    Arguments:
        model: A function that contains `mcx` model definition.
        rng_key: A PRNG key used to draw forward samples one at a time.

    Examples:

        Let us work with the following linear model to illustrate:

        >>> @mcx.model
        ... def linear_model(X):
        ...     weights @ Normal(0, 1)
        ...     sigma @ HalfNormal(0, 1)
        ...     y @ Normal(np.dot(X, weights), sigma)
        ...     return y
        ...
        ... mymodel = linear_model


        To perform forward sampling with a full input dataset, use the `sample`
        method. The `sample_shape` denotes the shape of the sample tensor to
        get *for each input data point*.

        >>> x = np.array([[1, 1], [2, 2], [3, 3]])
        ... samples = model.sample(X, sample_shape=(100, 10))
        ... print(samples.shape)
        (3, 100, 10)

        To explore the model, we can also use the "do" operator to fix the value of
        a random variable. This returns a copy of the model where all connections
        with the parent nodes have been removed:

        >>> conditioned = mymodel.do(sigma=0.2)
        ... conditioned.sample(rng_key, X, sample_shape=(1000,)  # single data point
        {'weights': 1.8, 'sigma': 0.2, 'y': 4.1}

        Unlike calling the model directly, this function is JIT-compiled and
        can be run on GPU. It should be quite fast to run, which should allow
        for quick iteration on the initial phase of the modeling process.

        You can also sample from the model's posterior by instantiating the `MCMC` class
        with the model:

        >>> sampler = MCMC(model)
        ... trace = sampler.run(X, y)

        Since `mcmx` only lightly alters python's AST, most of what you
        would do inside a function in Python is also valid in model
        definitions. For instance, complex transformations can be defined in
        functions that are called within the model. **It is required, however,
        that functions operate on and return only scalars or numpy arrays.***

        >>> from utils import mult
        ...
        ... @mcx.model
        ... def linear_model(X):
        ...     weights @ Normal(0, 1)
        ...     sigma @ HalfNormal(0, 1)
        ...     y @ Normal(mult(X, weights), sigma)
        ...     return y

        will work as the first example above.

        In the same vein, following PyMC4's philosophy [2]_, we would like to
        define complex distribution with the `mcmx` syntax:

        >>> @mcx.model
        ... def Prior():
        ...     return Normal(0, 1)
        ...
        ... @mcx.model
        ... def Horseshoe(tau0 , n):
        ...     tau @ HalfCauchy(0 , tau0)
        ...     delta @ HalfCauchy(0 , 1 , plate=n)
        ...     beta @ Normal (0, tau * delta )
        ...     return beta
        ...
        ... def my_model(X):
        ...     x @ Prior()
        ...     w @ Horseshoe(x, 0)
        ...     z = np.log(w)
        ...     y @ Normal(z, 1)

        Note the use of the decorator `@mcx.distribution`, which allows to make
        the distinction between generative models that interact with data, and
        derived distributions. Calling `model(Horseshoe)` reifies the
        distribution into a generative model from which we can generate forward
        samples and posterior samples.

    References:
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
        functools.update_wrapper(self, fn)

        # (TODO) Parse the source code here into intermediate representation
        # self.graph = parse(self.model_fn)

    def __call__(self) -> "model":
        return self

    def __print__(self):
        # random variables
        # deterministic variables
        # model source definition
        raise NotImplementedError

    def do(self, **kwargs) -> "model":
        """Apply the do operator to the graph and return a copy.

        The do-operator do(X=x) sets the value of the variable X to x and
        removes all edges between X and its parents. Applying the do-operator
        may be useful to analyze the behavior of the model pre-inference.

        References:
            Pearl, Judea. Causality. Cambridge university press, 2009.
        """
        for key in kwargs.keys():
            assert key in self.randvars

        conditionned_model = self.graph.do(kwargs)
        new_model = type("new_mode", (self,), {"graph": conditionned_model})
        return new_model

    def sample(self, *args, rng_key=None, sample_shape=(1000,)) -> numpy.ndarray:
        sampler = to_prior_sampler(self.model_fn, self.namespace)
        return sampler(rng_key, *args, sample_shape)

    def logpdf(self, *args) -> float:
        logpdf = to_logpdf(self.model_fn, self.namespace)
        return logpdf(*args)
