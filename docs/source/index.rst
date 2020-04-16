MCX: XLA-rated Bayesian inference
=================================

**mcx** is a Bayesian modeling language and inference library built on top of
JAX. **mcx** natively supports batching on GPU and TPU. It offers "turn key"
algorithms, but its inference engine is highly customizable.

------

**MCX by example**::
        
  import jax
  from jax import numpy as np
  import mcx
  import mcx.distributions as dist

  x_data = np.array([2.3, 8.2, 1.8])
  y_data = np.array([1.7, 7., 3.1])

  @mcx.model
  def linear_regression(x, lmbda=1.):
      scale @ dist.Exponential(lmbda)
      coefs @ dist.Normal(np.zeros(np.shape(x)[-1]))
      y = np.dot(x, coefs)
      predictions @ dist.Normal(y, scale)
      return predictions

  rng_key = jax.random.PRNGKey(0)

  # Sample the model forward, conditioning on the value of `x`
  mcx.sample_forward(
      rng_key,
      linear_regression,
      x=x_data,
      num_samples=10_000
  )
      
  # Sample from the posterior distribution using HMC
  kernel = mcx.HMC(
      step_size=0.01,
      num_integration_steps=100,
      mass_matrix_sqrt=np.array([1., 1.]),
      inverse_mass_matrix=np.array([1., 1.]),
  )
        
  observations = {'x': x_data, 'predictions': y_data, 'lmbda': 3.}
  sampler = mcx.sample(
      rng_key,
      linear_regression,
      kernel,
      **observations
  )
  trace = sampler.run()

The `HMC` kernel API seems convoluted for now; once warmup will be implemented,
every parameter besides `num_integration_steps` will be optional. Empirical HMC
is being implemented at the same time and all parameters will be optional in this
case.

Features
--------

* HMC and empirical HMC algorithms
* A dozen of distributions. More to come!
* Iterative inference
* Sequential inference
* Rhat, effective sample size and simulation based calibration
* Batch inference on GPU & TPU

mcx models are dynamic graphs that can be inspected and modified at runtime.


From practice to research
-------------------------

**mcx** is highly flexible and can be seen as the combination of two different libraries:

1. It can be used as a *modeling* language; mcx models can be exported as a
   forward sampling and a log probability density functions to be used with
   custom inference algorithms.
2. It can be used as an **inference** library; mcx's inference module is
   purposefully designed as a collection of loosely coupled elements. Advanced
   users are free to compose these elements in any way they see fit.

**mcx** strikes the right balance between customizability (for researchers) and
sane defaults for people who want the benefits of inference without having to
dig in the literature.

Quickstart
----------

Tutorials
---------

API reference
--------------

.. toctree::
   :maxdepth: 2

   api


Indices and tables
==================

* :ref:`search`
* :ref:`genindex`
* :ref:`modindex`
