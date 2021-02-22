# The implementation follows the design in Numpyro: numpyro.distributions.continuou
#
# Copyright (c) 2019-     The Numpyro project
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
from jax import lax
from jax import numpy as jnp
from jax import random, scipy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution
from mcx.distributions.shapes import promote_shapes


class MvNormal(Distribution):
    params_constraints = {
        "mu": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
    }
    support = constraints.real_vector

    def __init__(self, mu, covariance_matrix):

        if jnp.ndim(mu) < 1:
            mu = jnp.reshape(mu, (1,) + jnp.shape(mu))

        (mu_event_shape,) = jnp.shape(mu)[-1:]
        covariance_event_shape = jnp.shape(covariance_matrix)[-2:]
        if (mu_event_shape, mu_event_shape) != covariance_event_shape:
            raise ValueError(
                (
                    f"The number of dimensions implied by `mu` ({mu_event_shape}),"
                    "does not match the dimensions implied by `covariance_matrix` "
                    f"({covariance_event_shape})"
                )
            )

        mu = mu[..., jnp.newaxis]
        mu, covariance_matrix = promote_shapes(mu, covariance_matrix)
        self.event_shape = jnp.shape(covariance_matrix)[-1:]
        self.batch_shape = lax.broadcast_shapes(
            jnp.shape(mu)[:-2], jnp.shape(covariance_matrix)[:-2]
        )

        self.mu = mu[..., 0].squeeze()
        self.covariance_matrix = covariance_matrix.squeeze()

    def sample(self, rng_key, sample_shape=()):
        # random.multivariate_normal automatically adds the event shape
        # to the shape passed as argument.
        shape = sample_shape + self.batch_shape
        draws = random.multivariate_normal(
            rng_key, mean=self.mu, cov=self.covariance_matrix, shape=shape
        )

        return draws

    # no need to check on support ]-infty, +infty[
    def logpdf(self, x):
        return scipy.stats.multivariate_normal.logpdf(
            x, mean=self.mu, cov=self.covariance_matrix
        )
