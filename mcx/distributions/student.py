import jax.numpy as jnp
from jax import random, scipy

from mcx.distributions import constraints
from mcx.distributions.distribution import Distribution


class StudentT(Distribution):
    parameters = {"df": constraints.strictly_positive}
    support = constraints.real

    def __init__(self, df):
        self.event_shape = ()
        self.batch_shape = jnp.shape(df)
        self.df = df

    def sample(self, rng_key, sample_shape=()):
        shape = sample_shape + self.batch_shape + self.event_shape
        return random.t(rng_key, self.df, shape)

    def logpdf(self, x):
        return scipy.stats.t.logpdf(x, self.df)
