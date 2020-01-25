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

import jax.numpy as np


__all__ = [
    "boolean",
    "interval",
    "integer",
    "integer_interval",
    "positive_integer",
    "positive",
    "probability",
    "real",
    "simplex",
]


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
        return x > self.lower_bound

    def __str__(self):
        return "a real number > {}".format(self.lower_bound)


class _Interval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __str__(self):
        return "a real number in [{},{}]".format(self.lower_bound, self.upper_bound)

    def __call__(self, x):
        return (x > self.lower_bound) & (x < self.upper_bound)


class _Integer(Constraint):
    def __call__(self, x):
        return x == np.floor(x)

    def __str__(self):
        return "an integer"


class _IntegerGreaterThan(Constraint):
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __str__(self):
        return "an integer > {}".format(self.lower_bound)

    def __call__(self, x):
        return (x == np.floor(x)) & (x > self.lower_bound)


class _IntegerInterval(Constraint):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __str__(self):
        return "an integer in [{}, {}]".format(self.lower_bound, self.upper_bound)

    def __call__(self, x):
        return (x == np.floor(x)) & (x >= self.lower_bound) & (x <= self.upper_bound)


class _Real(Constraint):
    def __str__(self):
        return "a real number"

    def __call__(self, x):
        return np.isfinite(x)


class _Simplex(Constraint):
    def __str__(self):
        return "a vector of numbers that sums to one up to 1e-6 (probability simplex)"

    def __call__(self, x):
        x_sum = np.sum(x, axis=-1)
        return np.all(x > 0, axis=-1) & (x_sum <= 1) & (x_sum > 1 - 1e-6)


boolean = _Boolean()
integer = _Integer()
integer_interval = _IntegerInterval
interval = _Interval
positive_integer = _IntegerGreaterThan(0)
positive = _GreaterThan(0.0)
probability = _Interval(0.0, 1.0)
real = _Real()
simplex = _Simplex()
