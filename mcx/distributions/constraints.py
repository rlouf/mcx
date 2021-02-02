# The implementation follows the design in PyTorch: torch.distributions.constraints.py
# and the modifications made in Numpyro: numpyro.distributions.constraints
#
# Copyright (c) 2019-     The Numpyro project
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from abc import ABC, abstractmethod

from jax import numpy as jnp

__all__ = [
    "limit_to_support",
    "boolean",
    "closed_interval",
    "interval",
    "integer",
    "integer_interval",
    "positive_integer",
    "positive",
    "probability",
    "real",
    "simplex",
    "strictly_positive",
    "real_vector",
]


# Sourced from numpyro.distributions.utils.py
# Copyright Contributors to the NumPyro project.
# SPDX-License-Identifier: Apache-2.0
def limit_to_support(logpdf):
    """Decorator that enforces the distrbution's support by returning `-jnp.inf`
    if the value passed to the logpdf is out of support.

    """

    def wrapper(self, *args):
        log_prob = logpdf(self, *args)
        value = args[0]
        mask = self.support(value)
        log_prob = jnp.where(mask, log_prob, -jnp.inf)
        return log_prob

    return wrapper


# ---------------------------------------------------------
#                  == CONSTRAINTS ==
# ---------------------------------------------------------


class Constraint(ABC):
    @abstractmethod
    def __call__(self, x):
        pass


class _Boolean(Constraint):
    def __call__(self, x):
        return (x == 0) | (x == 1)

    def __str__(self):
        return "a boolean"


class _GreaterThan(Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, x):
        return x >= self.lower_bound

    def __str__(self):
        return f"a real number >= {self.lower_bound}"


class _StrictlyGreaterThan(Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, x):
        return x > self.lower_bound

    def __str__(self):
        return f"a real number > {self.lower_bound}"


class _Interval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __str__(self):
        return f"a real number in ]{self.lower_bound},{self.upper_bound}["

    def __call__(self, x):
        return (x > self.lower_bound) & (x < self.upper_bound)


class _ClosedInterval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __str__(self):
        return f"a real number in [{self.lower_bound},{self.upper_bound}]"

    def __call__(self, x):
        return (x >= self.lower_bound) & (x <= self.upper_bound)


class _Integer(Constraint):
    def __call__(self, x):
        return x == jnp.floor(x)

    def __str__(self):
        return "an integer"


class _IntegerGreaterThan(Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __str__(self):
        return f"an integer > {self.lower_bound}"

    def __call__(self, x):
        return (x == jnp.floor(x)) & (x > self.lower_bound)


class _IntegerInterval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __str__(self):
        return f"an integer in [{self.lower_bound},{self.upper_bound}]"

    def __call__(self, x):
        return (x == jnp.floor(x)) & (x >= self.lower_bound) & (x <= self.upper_bound)


class _Real(Constraint):
    def __str__(self):
        return "a real number"

    def __call__(self, x):
        return jnp.isfinite(x)


class _Simplex(Constraint):
    def __str__(self):
        return "a vector of numbers that sums to one up to 1e-6 (probability simplex)"

    def __call__(self, x):
        x_sum = jnp.sum(x, axis=-1)
        return jnp.all(x > 0, axis=-1) & (x_sum <= 1) & (x_sum > 1 - 1e-6)


class _IndependentConstraint(Constraint):
    def __str__(self):
        return "an event is valid only if all its independent entries are valid."

    def __init__(self, base_constraint, ndims):
        assert isinstance(base_constraint, Constraint)
        assert isinstance(ndims, int)
        assert ndims >= 0

        self.base_constraint = base_constraint
        self.ndims = ndims
        super().__init__()

    def __call__(self, value):
        result = self.base_constraint(value)
        if self.ndims == 0:
            return result
        elif jnp.ndim(result) < self.ndims:
            raise ValueError(
                (f"Expected value.dim() >= {self.ndims}" " but got {jnp.ndims{result}}")
            )
        result = result.all(-1)
        return result


class _PositiveDefinite(Constraint):
    event_dim = 2

    def __str__(self):
        return "a positive definite matrix."

    def __call__(self, x):
        symmetric = jnp.all(jnp.all(x == jnp.swapaxes(x, -2, -1), axis=-1), axis=-1)
        # check for the smallest eigenvalue is positive
        positive = jnp.linalg.eigh(x)[0][..., 0] > 0
        return symmetric & positive


boolean = _Boolean()
closed_interval = _ClosedInterval
integer = _Integer()
integer_interval = _IntegerInterval
interval = _Interval
positive_integer = _IntegerGreaterThan(0)
positive = _GreaterThan(0.0)
positive_definite = _PositiveDefinite()
probability = _Interval(0.0, 1.0)
real = _Real()
real_vector = _IndependentConstraint(real, 1)
simplex = _Simplex()
strictly_positive = _StrictlyGreaterThan(0.0)
