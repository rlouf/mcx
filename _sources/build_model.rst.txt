===================
Build models in MCX
===================

Syntax
------

Bayesian models are often called *generative models* since they can generate
random outcomes. MCX adheres to this description: models are functions that
return a random value each time they are called. 

MCX models thus look pretty much like any Python function. With two important
particularities:

1. To signal that a function is a model it must be preceded with the
   `@mcx.model` decorator. Otherwise Python will interpret it as a regular
   function and it will not work.
2. MCX uses the symbol `<~` for random variable assignments and `=` for
   deterministic assignments. As a result models are visually very similar
   to their mathematical counterpart.

To illustrate this, let us model the number of successes in `N` tosses of a
coin:

.. code-block:: python

   import mcx
   from mcx.distributions import Beta, Binomial

   @mcx.model
   def coin_toss(N):
       p <~ Beta(.5, .5)
       successes <~ Binomial(N, p)
       return successes

As we said, generative models behave like functions:

>>> coin_toss(10)
4
>>> coin_toss(10)
7

Since the parameters are random variables, each call will return a different
value. If you want to generate a large number of values, you can simply iterate:

>>> value = [coin_toss(10) for _ in range(100)]


Caveats
*******

The MCX language is still young and comes with a few caveats, things that you
cannot do when expressing a model. As time passes, code is written and PRs are
merged these constraints will be relaxed and you will be able to written MCX
code like you would regular python code.

First any variable, random or deterministic, must be given a name. The
following functions are invalid MCX code:

.. code-block:: python

    @mcx.model
    def return_value_not_assigned():
        """The returned variable must have a name."""
        a <~ Normal(0, 1)
        b <~ Gamma(1, a)
        return a * b

    @mcx.model
    def random_argument_not_assigned():
        """Normal(0, 1) must have an explicit name."""
        b <~ Gamma(1, Normal(0, 1))
        return b

    @mcx.model
    def deterministic_expression_not_assigned():
        """You cannot use deterministic expressions as an argument.
        """
        a <~ Normal(0, 1)
        b <~ Gamma(1, np.exp(a))
        return b

Control flow is also not supported for the moment. MCX will not compile
functions such as:

.. code-block:: python

    @mcx.model
    def if_else():
        a <~ Bernoulli(.5)

        if a > .3:
            b <~ Normal(0, 1)
        else:
            b <~ Normal(0, 2)

        return b

    @mcx.model
    def for_loop():
        a <~ Poisson(10)
        
        total = 0
        for i in range(1, a):
            b <~ Bernoulli(1./i)
            total += b

        return total

This will eventually be integrated to support models with stochastic support.
Model closures are also currently not supported. MCX will not process code such
as:

.. code-block:: python

    @mcx.model
    def closure():
        p <~ Beta(.5, .5)

        @mcx.model
        def toss():
            is_head <~ Binomial(p)
            return is_head

        return toss

These will eventually be supported as part of the implementation of non-parametrics.

Call functions from a model
---------------------------

You can call other (deterministic) python functions inside a generative model
as long as they only use python operators or functions implemented in
`JAX <https://jax.readthedocs.io/en/latest/jax.html>`_ (most of numpy's and some of
scipy's methods).

.. code-block:: python

  import mcx
  from mcx.distributions import Exponential, Normal

  def multiply(x, y):
      return x * y

  @mcx.model
  def one_dimensional_linear_regression(X):
        sigma <~ Exponential(.3)
        mu <~ Normal(np.zeros_like(X))
        y = multiply(X, mu)
        return Normal(y, sigma)


Models are (multivariate) distributions
----------------------------------------

Most distributions can be seen as the result of a generative process. For
instance you can re-implement the exponential distribution in MCX as

.. code-block:: python
  
  import jax.numpy as np
  import mcx
  from mcx.distributions import Exponential
        
  @mcx.model
  def Exponential(lmbda):
      U <~ Uniform(0, 1)
      t = - np.log(U) / lmbda
      return t

When we say that we "sample" from the exponential distribution, we are actually
interested in the value of `t`, discarding the values taken by `u`.

By analogy, a generative model expressed with MCX can also be used as a
distribution, which is the distribution of the returned value. It is thus
possible to compose MCX models as follows

.. code-block:: python

  import mcx
  from mcx.distributions import HalfCauchy, Normal

  @mcx.model
  def HorseShoe(mu, tau, s):
      scale <~ HalfCauchy(0, s)
      noise <~ Normal(0, tau)
      h = mu + scale * noise
      return h

  @mcx.model
  def one_dimensional_linear_regression(X):
      sigma <~ Exponential(.3)
      mu <~ HorseShoe(np.zeros_like(X), 1., 1.)
      z = X * mu
      y <~ Normal(mu, sigma)
      return y

Which encourages code re-use and modularity.


Querying / Debugging the model
-------------------------------

MCX translates model definitions in an intermediate representation (a graph)
which can be dynamically queried and modified at runtime. Three features of MCX
make debugging a model easier: forward sampling, conditioning and the ability to
modify the model dynamically.

Forward sampling
================

Forward sampling means sampling from the prior distribution of each variable in
the model. We sample for one data point at a time or the whole dataset in one
go.

.. code-block:: python
        
  import jax
  import mcx
        
  rng_key = jax.random.PRNGKey(0)
  mcx.sample_forward(rng_key, model, **data)

Intervention
============

Sometimes we want to set the value of a variable in the model to a constant. We
can do so using the `do` operator which can be combined with the `sample_forward`
function:

.. code-block:: python
        
  import jax
  import mcx
        
  rng_key = jax.random.PRNGKey(0)
  model_c = model.do(rv=10)
  mcx.sample_forward(rng_key, model_c, **data)


Modify the graph dynamically
============================

We can also inspect the values of variables individually

>>> model["y"]

You can also change the variables' distribution for quick iteration on the
model:

>>> model['y'] = "Normal(0, 10)"

This is pretty much all there is to known about how to express models with MCX.
You can consult the section of the documentation on distributions to get
acquainted with the distributions included with MCX.
