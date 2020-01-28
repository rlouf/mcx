from typing import Callable, Any, Optional
from types import FunctionType

import numpy
import jax

from mcx.compiler import to_graph, to_logpdf, to_prior_sampler
from mcx.distributions import Distribution


class model(Distribution):
    """Representation of a model.

    Since it represents a probability graphical model, the `model` instance is
    a probability distribution, and as such inherits from the `Distribution`
    class. It implements the `sample` and `logpdf` methods.

    Models are expressed as functions. The expression of the model within the
    function should match the way it would be expressed mathematically as
    possible. The only difference with standard python code is the use of
    the "@" operator for random variable assignments.

    The models are then parsed into an internal representation that can be
    conditioned on data, compiled into a logpdf or a forward sampler. The
    result is pure functions that can be further JIT-compiled with JAX,
    differentiated and dispatched on GPUs and TPUs.

    Arguments:
        model: A function that contains `mcx` model definition.
        rng_key: A PRNG key used to draw forward samples one at a time.

    Examples:

        Let us work with the following linear model to illustrate:

        >>> def linear_model(X):
        ...     weights @ Normal(0, 1)
        ...     sigma @ HalfNormal(0, 1)
        ...     y @ Normal(np.dot(X, weights), sigma)
        ...     return y
        ...
        ... mymodel = model(linear_model)

        We can call the model with the data and get samples. It is not clear at
        the moment whether I should only return samples from `y` or every other
        weight too.

        >>> mymodel(X)  # single data point
        {'weights': 1.8, 'sigma': 0.1, 'y': 3.7}
        ... mymodel(X)
        -3.4

        To explore the model, we can also use the "do" operator to fix the value of
        a random variable. This returns a copy of the model where all connections
        with the parent nodes have been removed:

        >>> conditioned = mymodel.do(sigma=0.2)
        ... print(conditioned(X))  # single data point
        {'weights': 1.8, 'sigma': 0.2, 'y': 4.1}

        To perform forward sampling with a full input dataset, use the `sample`
        method. The `sample_shape` denotes the shape of the sample tensor to
        get *for each input data point*.

        >>> x = np.array([[1, 1], [2, 2], [3, 3]])
        ... samples = model.sample(X, sample_shape=(100, 10))
        ... print(samples.shape)
        (3, 100, 10)

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
        ... def linear_model(X):
        ...     weights @ Normal(0, 1)
        ...     sigma @ HalfNormal(0, 1)
        ...     y @ Normal(mult(X, weights), sigma)
        ...     return y

        will work as the first example above.

        In the same vein, following PyMC4's philosophy [2]_, we can define complex
        distribution with the `mcmx` syntax:

        >>> @mcx.distribution
        ... def Prior():
        ...     return Normal(0, 1)
        ...
        ... @mcx.distribution
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

    def __init__(self, model: FunctionType, rng_key: Optional[jax.random.PRNGKey] = None) -> None:
        self.model_fn = model
        self.namespace = model.__globals__

        self.graph = to_graph(model)

        self.rng_key = rng_key
        if rng_key is None:
            self.rng_key = jax.random.PRNGKey(0)

    def __call__(self, *args, rng_key=None) -> numpy.ndarray:
        _, self.rng_key = jax.random.split(self.rng_key)
        return self.sample(*args, rng_key=self.rng_key, sample_shape=(1,))

    def sample(self, *args, rng_key=None, sample_shape=(1000,)) -> numpy.ndarray:
        sampler = to_prior_sampler(self.model_fn, self.namespace)
        return sampler(rng_key, *args, sample_shape)

    def logpdf(self, *args) -> float:
        logpdf = to_logpdf(self.model_fn, self.namespace)
        return logpdf(*args)
