import mcx.distributions as md
import trax.layer as tl


class Layer(tl.Layer, md.Distribution):
    """Base class for composable layers in a Bayesian deep model.

    Random layers are the building block of Bayesian deep models. We implement
    sublayers by subclassing trax's layers. We denote here by 'random layers'
    layers whose weights are drawn from a distribution. To define a random
    layer, one must use the `@` symbol to indicate a random variable.

    Random layers are distributions on the space of functions, and as such must
    behave like a Distribution: each layers must be associated with a
    log-probability density function and a function that returns samples from
    the layer's distribution.

    Here are the functionalities we want to implement before a first release:

    - [ ] ml.Serial construct that takes care of `logpdf` and `sample`
    - [ ] dense layer
    - [ ] softmax
    - [ ] ReLu
    - [ ] Figure broadcasting out
    - [ ] Deterministic transformations
    - [ ] Handle the case where defining `Normal(0, 1)` in the NN definition and
          outside in the example below return the same thing.

    Examples
    --------

    We can define a simple neural network with one hidden layer to classify
    MNIST images as:

    >>> @mcx.model
    ... def mnist(image):
    ...     nn @ ml.Serial(
    ...         dense(400, Normal(0, 1)),
    ...         dense(400, Normal(0, 1)),
    ...         dense(10, Normal(0, 1)),
    ...         softmax(),
    ...     )
    ...     probs = nn(image)
    ...     cat = Categorical(probs)
    ...     return cat

    One can easily add a hierarchical structure on top of layers:

    >>> @mcx.model
    ... def mnist(image):
    ...     s @ Poisson(.5)
    ...     nn @ ml.Serial(
    ...         dense(400, Normal(0, s)),
    ...         dense(400, Normal(0, s)),
    ...         dense(10, Normal(0, s)),
    ...         softmax(),
    ...     )
    ...     probs = nn(image)
    ...     cat @ Categorical(probs)
    ...     return cat

    In case we want a more complex probabilistic structure of the weights,
    we can pass weight transformations to the layers:

    >>> @mcx.model
    ... def mnist(image):
    ...     b1 @ Bernoulli(.5)
    ...     nn @ ml.Serial(
    ...         dense(400, Normal(0, 1), fn=lambda w: w*b1),
    ...         dense(400, Normal(0, 1)),
    ...         dense(10, Normal(0, 1)),
    ...         softmax(),
    ...     )
    ...     probs = nn(image)
    ...     cat @ Categorical(probs)
    ...     return cat

    """

    def logpdf(self, x):
        """Returns the logpdf associated with the layer. Defaults to 0 for
        deterministic layers.
        """
        return 0

    def sample(self):
        return self.forward(self.weights)
