=================================
MCX: XLA-rated Bayesian inference
=================================

**Intuitive modeling language**
MCX is a bayesian modeling language and inference library built on top of
JAX. Expressing a generative model in MCX is as simple as being able to
write it down on a piece of paper and having some basic knowledge of Numpy.

**Performant inference**
MCX natively supports batching on GPU for faster inference and sequential
inference for bayesian updating.

**The best of usability and modularity**
MCX reconciles modularity with usability. It does offer "turn key" algorithms
for users who just want something that works. But it also allows users to
compose their own inference kernel from multiple parts.


MCX by example
==============

.. code-block:: python

  import jax
  from jax import numpy as np
  import mcx
  import mcx.distributions as dist


  rng_key = jax.random.PRNGKey(0)
  X, y = observations()

  @mcx.model
  def linear_regression(x, lmbda=1.):
      scale <~ dist.Exponential(lmbda)
      coefs <~ dist.Normal(np.zeros(np.shape(x)[-1]))
      y = np.dot(x, coefs)
      predictions <~ dist.Normal(y, scale)
      return predictions

  # Sample from the model posterior distribution using HMC    
  hmc_kernel = mcx.HMC(
      step_size=0.01,
      num_integration_steps=100,
      mass_matrix=np.array([1., 1.])
  )
        
  observations = {'x': X, 'predictions': y, 'lmbda': 3.}
  sampler = mcx.sample(rng_key, linear_regression, hmc_kernel, **observations)
  trace = sampler.run()


Features
========

* The HMC and empirical HMC algorithms;
* A dozen distributions;
* Batch sampling;
* Iterative sampling for more complex workflows;
* Sequential inference for bayesian updating;
* Sample millions of chains in parallel;
* Fast inference on GPU.


From practice to research
=========================

MCX is highly flexible and can be seen as the combination of two decoupled
libraries:

1. *A probabilistic modeling language*; MCX models can be exported as a forward
   sampling function and a log-probability density functions which may be used
   with custom inference algorithms.
2. *An inference library*; MCX's inference module is purposefully designed as a
   collection of loosely coupled elements. Advanced users are free to compose
   these elements in any way they see fit. It is not necessary to use the
   modeling language to build the log-probability density function, any python
   function will do. 

MCX tries to strike the right balance between customizability (for researchers)
and sane defaults for people who want the benefits of inference without having
to dig in the literature.


Quickstart
==========

.. toctree::
    :maxdepth: 1
        
    build_model
    prior_predictive
    inference
    diagnostics
    posterior inference
