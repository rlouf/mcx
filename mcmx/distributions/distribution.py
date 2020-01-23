from abc import ABC, abstractmethod
from typing import Dict, Tuple

import jax

from .constraints import Constraint


class Distribution(ABC):
    """Represents a probability distribution.

    A distribution is an object that can generate samples and to which a
    log-probability distribution function (logpdf) is associated.

    Keeping track of shapes during sampling is no easy task _[1]; we follow
    Tensorflow Distribution's shape system _[2,3] which decomposes shapes along
    three dimensions:

    - *sample_shape:* independent, identically distributed samples from the same
      distribution;
    - *batch_shape:* independent samples from different distributions;
    - *event_shape:* shape of a simple draw from a distribution;

    Each distribution is defined on a support, and trying to compute the logpdf
    outside of the support should return -infinity. We generally follow the
    design of Pytorch Distributions _[4] and define a set of constraints that
    return `True` when the input satisfies the contraint, `False` otherwise,
    and assign a constraint to the support. The `logpdf` method is wrapped by a
    decorator that checks whether arguments belong to the support.

    Attributes
    ----------
    params_constraints: Dict
        The constraints on the values of the parameters.
    support: Type[Constraint]
        The support of the logpdf.
    sample_shape: Tuple
        Describes independent, identically distributed samples from the same distribution _[TFD].
    batch_shape: Tuple
        Describes independant samples from different distributions _[TFD].
    event_shape: Tuple
        Shape of a single draw from the distribution _[TFD].

    References
    ----------
    ..[1] Luciano Paz. (2019) PyMC3 shape handling.
          https://lucianopaz.github.io/2019/08/19/pymc3-shape-handling/
    ..[2] Dillon, J. V., Langmore, I., Tran, D., Brevdo, E., Vasudevan, S.,
          Moore, D., ... & Saurous, R. A. (2017). Tensorflow distributions.
          arXiv preprint arXiv:1711.10604. (section 3.3)
    ..[3] Eric J. Ma. (2019) Reasoning about shapes and probability distributions.
          https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/
    ..[4] Pytorch Distributions. https://pytorch.org/docs/stable/distributions.html
    """

    param_constraints: Dict[str, Constraint]
    support: Constraint

    @abstractmethod
    def __init__(self, *args) -> None:
        pass

    @abstractmethod
    def sample(
        self, rng_key: jax.random.PRNGKey, sample_shape: Tuple[int]
    ) -> jax.numpy.DeviceArray:
        """Obtain samples from the distribution.

        Arguments
        ---------
        rng_key: jax.random.PRNGKey
            The pseudo random number generator key to use to draw samples.
        sample_shape: Tuple[int]
            The number of independant, identically distributed samples to draw
            from the distribution.

        Returns
        -------
        jax.numpy.DeviceArray
            An array of shape sample_shape + batch_shape + event_shape with independent samples.
        """
        pass

    @abstractmethod
    def logpdf(self, x: jax.numpy.DeviceArray) -> jax.numpy.DeviceArray:
        """Compute the value of the log-probability density function at a given
        point.

        Unlike PyTorch or Numpyro, the value's legal status is not checked
        dynamically but at compile time.

        **Important**

        Be careful when reasoning about the broadcasting behaviour of the
        logpdf function! Consider the following example with a Normal
        distribution and batch dimension 2:

            >>> x = np.array([1, 3])
            ... log_prob = Normal(mu=np.array([0, 1]), sigma=1).logpdf(x)
            ... log_prob.shape
            (2,)

        `jax.numpy` did broadcast (as it should per broadcasting rules), which
        is the *expected* behavior. The *desired* behavior is different: when
        the event shape is different from 0, it is assumed that the values in a
        `x` array should correspond to different proposals, and the output of
        the logpdf would be a (2,2) array.

        We shall thus asume in this library that `logpdf` computes the log
        probability value for ONE proposal. To get the values for multiple
        proposals, the computation should be vectorized using `jax.vmap`.

        Arguments
        ---------
        x: jax.numpy.DeviceArray, shape (n_points,)
            The point(s) at which to evaluate the log probability density function.

        Returns
        -------
        jax.numpy.DeviceArray, shape (n_points,)
            The value(s) of the log-probability density function.
        """
        pass

    def __str__(self):
        """User-friendly representation of the probability distribution.
        """
        constraints_str = "\n  ".join(
            ["{0}: {1!s}".format(key, value) for key, value in self.params_constraints.items()]
        )
        support_str = str(self.support)
        return (
            "{} distribution".format(self.__class__.__name__)
            + "\n  batch_shape: {}".format(self.batch_shape)
            + "\n  event_shape: {}".format(self.event_shape)
            + "\n\nParameters\n  {}".format(constraints_str)
            + "\n\nSupport: {}\n".format(support_str)
        )
