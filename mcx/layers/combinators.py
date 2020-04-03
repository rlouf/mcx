import mcx.distributions as md

try:
    import trax.layers as tl
except ImportError:
    raise ImportError(
        "You need to install trax (`pip install trax`) to use the neural networks functionalities."
    )


class Serial(tl.Serial, md.Distribution):
    """Combinator that applies layers sequentially.

    >>> def mnist(image):
    ...     nn @ ml.Serial(
    ...         dense(400, Normal(0, 1)),
    ...         dense(400, Normal(0, 1)),
    ...         dense(10, Normal(0, 1)),
    ...         softmax(),
    ...     )
    ...     p = nn(image)
    ...     cat @ Categorical(p)
    ...     return cat

    Computing the logpdf:

    >>> def mnist_logpdf(image, w1, w2, w3, cat):
    ...    nn_logpdf = nn.logpdf(w1, w2, w3)
    ...    # set layerweights to w1, w2, w3
    ...    p = nn(image)
    ...    cat_logpdf = Categorical(p).logpdf(cat)
    ...    return nn_logpdf + cat_logpdf

    Sampling from the distribution:

    >>> def mnist_sample(rng_key, image, shape=()):
    ...     nn = ml.Serial(
    ...         dense(400, Normal(0, 1)),
    ...         dense(400, Normal(0, 1)),
    ...         dense(10, Normal(0, 1)),
    ...         softmax(),
    ...     ).sample(rng_key, shape)
    ...     p = nn(image)
    ...     cat @ Categorical(p).sample(rng_key, shape)
    ...     return cat

    """

    def __init__(self, *sublayers):
        super(tl.Serial, self).__init__()

    def logpdf(self, *weights):
        """Should compute the logpdf of the weights as well
        as setting the weights to the passed value.

        Example
        -------

        The following model:

        >>> @mcx.model
        ... def prior():
        ...     nn @ ml.Serial(
        ...         dense(10, Normal(0, 1)),
        ...         relu(),
        ...         dense(20, Normal(0, 1)),
        ...         softmax(),
        ...     )
        ...     return nn

        Should be associated with the following logpdf function:

        >>> def prior_logpdf(w1, w2):
        ...     logpdf += Normal(0, 1).logpdf(w1)
        ...     logpdf += Normal(0, 1).logpdf(w2)
        ...     return logpdf

        Which is obtained by summing the log-probability associated with
        each layer.
        """
        logpdf = 0
        for layer, w in zip(self.sublayers, weights):
            logpdf += layer.logpdf(w)

    def sample(self, rng_key, sample_shape):
        """Should make sure that the forward pass generates samples
        every time a new element is passed as an input.

        >>> nn.sample(rng_key, sample_shape)

        Should return a sample of the `nn` function; this sets the value
        of the weights for the next forward pass.

        -> Overloads forward() ? i.e. returns a function.
           In which case, to be consistent, we should
           implement forward() for our model as getting a sample of
           the generated variable.
        """
        self.rng_key = rng_key
        self.sample_shape = sample_shape
        self.forward_with_state = self.sample_with_state
        return self
