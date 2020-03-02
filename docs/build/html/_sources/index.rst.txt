MCX: XLA-rated Bayesian inference
=================================

**mcx** is a Bayesian modeling language and inference library built on top of
JAX. **mcx** natively supports batching on GPU and TPU. It offers "turn key"
algorithms, but its inference engine is highly customizable.

------

**MCX by example**::
        
  from jax import numpy as np
  import mcx
  import mcx.distributions as dist

  x_data = np.array([2.3, 8.2, 1.8])
  y_data = np.array([1.7, 7., 3.1])

  @mcx.model
  def linear_regression(x, lmbda=1.):
      scale = dist.Exponential(lmbda)
      coefs = dist.Normal(np.zeros(np.shape(x)[-1]))
      y = np.dot(x, coefs)
      predictions = dist.Normal(y, scale)
      return predictions

  # Forward (or prior-predictive) samples
  mcx.sample_forward(
      linear_regression,
      x=data['x'],
      num_samples=10_000
  )
      
  # Posterior samples
  kernel = mcx.dynamicHMC(model)
  samples = mcx.sample(
      kernel,
      x=data['x'],
      predictions=data['y'],
      lmbda=3.,
      num_samples=1000
  )


Features
--------

* dynamic HMC (NUTS) and empirical HMC algorithms
* A dozen of distributions. More to come!
* Sequential inference
* Iterative inference
* Rhat, effective sample size and simulation based calibration
* mcx models are dynamic graphs that can be inspected and modified at runtime.
* Batch inference on GPU & TPU


From practice to research
-------------------------

**mcx** is highly flexible and can be seen as the combination of two different libraries:

1. It can be used as a *modeling* language; mcx models can be exported as a
   forward sampling and a log probability density functions to be used with
   custom inference algorithms.
2. It can be used as an inference library; mcx's inference module is
   purposefully designed as a collection of loosely coupled elements. Advanced
   users are free to compose these elements in any way they see fit. Warmup is
   also very customizable.

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
