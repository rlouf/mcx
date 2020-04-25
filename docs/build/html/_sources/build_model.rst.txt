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
   to their mathematical counterpart. It also signals that the variables 
   assigned are not just containers for a value, that this relationship has
   different meaning depending on the context.

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

Generative models behave like functions:

>>> coin_toss(10)
4
>>> coin_toss(10)
7

Since the parameters are random variables, each call will return a different
value. If you want to generate a large number of values, you can simply iterate:

>>> value = [coin_toss(10) for _ in range(100)]


Caveats
*******

MCX comes with a few caveats, things that you cannot do when expressing a
model. First any variable, random or deterministic, must be assigned to a name.
The following functions are invalid MCX code:

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
    def control_flow():
        a <~ Bernoulli(.5)

        if a > .3:
            b <~ Normal(0, 1)
        else:
            b <~ Normal(0, 2)

        return b

Its implementation is currently on the roadmap.


Call functions from a model
---------------------------

You can call other (deterministic) python functions inside a generative
model as long as they only use python operators or functions implemented
in `JAX <https://jax.readthedocs.io/en/latest/jax.html>`_.

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


Use models as distributions 
---------------------------

Generative functions implicitely define a multivariate distribution. They
"augment" this distribution with an execution model and define some variables
as inputs and outputs.

Conversely, most distributions can also be seen as the result of a generative
process. For instance you can re-implement the exponential distribution in MCX
as

.. code-block:: python
  
  import jax.numpy as np
  import mcx
  from mcx.distributions import Exponential
        
  @mcx.model
  def Exponential(lmbda):
      U <~ Uniform
      t = - np.log(U) / lmba
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

MCX translates model definitions in an intermediate representation which can be
dynamically queried and modified at runtime.

It is possible to interact dynamically with a model. Let us use the
one-dimensional linear regression example. First you can query the
prior values for each variable.

>>> model = one_dimensional_linear_regression(np.array([1., 2., 3.])
>>> model['y']
... shows information about the distibution, shape, etc.

You can also change the variables' distribution for quick iteration on the
model:

>>> model['y'] = "Normal(0, 10)"

This is pretty much all there is to known about how to express models with MCX.
You can consult the section of the documentation on distributions to get
acquainted with the distributions included with MCX. You can also have a look
at the API for neural network layers.

Next we will see how to sample from the model's prior distribution.
