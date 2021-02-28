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
for users who just want something that works. Or you can go your own way, and use MCX's expressive modeling language to compile a model's logpdf as a python function. Which you can then use with your algorithms or those of libraries like `BlackJAX <https://github.com/blackjax-devs/blackjax>`_.


MCX by example
==============

.. code-block:: python

  import jax
  from jax import numpy as jnp
  import mcx
  import mcx.distributions as dist


  rng_key = jax.random.PRNGKey(0)
  X, y = observations()

  @mcx.model
  def linear_regression(x, lmbda=1.):
      scale <~ dist.Exponential(lmbda)
      coefs <~ dist.Normal(jnp.zeros(jnp.shape(x)[-1]))
      y = jnp.dot(x, coefs)
      predictions <~ dist.Normal(y, scale)
      return predictions

  # Sample from the model posterior distribution using HMC    
  hmc_kernel = mcx.HMC(num_integration_steps=100)
        
  args = (X, 3.)
  observations = {'predictions': y}
  sampler = mcx.sample(
     rng_key, 
     linear_regression,
     args,
     observations,
     hmc_kernel
  )
  trace = sampler.run()


Features
========

* HMC and NUTS algorithms with window adaptation;
* A dozen distributions;
* Batch sampling;
* Iterative sampling for more complex workflows;
* Sample hundreds of thousands of chains in parallel;
* Fast inference on GPU.


Quickstart
==========

.. toctree::
    :maxdepth: 1
        
    build_model
    inference
