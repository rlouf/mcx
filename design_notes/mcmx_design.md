# MCMX design

## Philosophy

We want a probabilistic programming library that

1. Is **expressive enough** to allow the user to write a large range of
   probabilistic programs, and sticks to the mathematical syntax as much as
   possible. Knowing how to express a probabilistic graphical model and
   manipulating Numpy arrays should be enough. It should provide a "no surprise"
   API.
2. Leverages **modern hardware** for inference, namely GPUs and TPUs. Inference
   should be performant to allow **fast iteration** in the exploration phase.
3. **Focuses exclusively on sampling** to approximate the posterior.
4. Welcomes **sequential sampling** as a first-class citizen.

## Probabilistic models are programs

Our understanding is that Probabilistic Programming libraries that rely on a
non-standard interpretation of a programming language need to somehow bridge the
gap between this langage's construct and the requirement of probabilistic models
and their use. Mainly:

- Random variable definitions can have two different intepretations depending on
  the contex. `x ~ Normal(0, 1)` can either mean "x points to the value of a
  random draw from the standard normal distribution" or "a (log) probability
  associated with the value of x: `Normal(0, 1).logpdf(x)`. There is no simple
  construct in Python to express these two meanings in one statement.
- Assuming we can translate models accurately, querying this representation
  presents challenges in itself. Different samplers have different requirements,
  and the random variables may need to be transformed depending on their
  support, but also purely to ease convergence.

The philosophy behind mcmx is simple:

> Probabilistic models are programs. Samplers are execution engines.

Like any programming language, mcmx has a specific syntax (extremely close to
that of Python) that allows to express programs, and a "compiler" (note: let me
know if there is a better way to call this) that transforms the program into a
way that is understandable by the execution engine. This compiler can perform
engine-specific optimizations.

```
                           +----------------+
                           |Forward sampling|
                           +----------------+
                                   ^                 +-------------------+
                                   |           +---->|Posterior sampler 1|
                                   |           |     +-------------------+
+----------------+             +--------+      |     +-------------------+
|Model definition|------------>|Compiler|----------->|Posterior sampler 2|
+----------------+             +--------+      |     +-------------------+
                                               |     +-------------------+
                                               +---->|Posterior sampler 3|
                                                     +-------------------+

```

**This idea is not new.** It is at the hear of languages such as Stan, Bugs,
JAGS, Anglican, HackPPL, Joss.jl and many that I am unfortunately not aware of.
The only innovation of `mcmx` is to apply this idea to a python library.

**There is a huge caveat.** Source code transformation/generation implies **a
lot of black magic** under the hood. If this is not done carefully, at best the
code will fail in unexpected way, make translation mistakes that affect the
inference.  
However, these issues can be mitigated by (1) staying as close to the Python
syntax as possible to benefit from the decades of development around its
Abstract Syntax Tree (2) Providing the generated source code to the user for
inspection.

Since the compiler generates python functions, we can further use JAX's
just-in-time compilation, autodiff and vectorization capabilities easily. Which
not only allows for performant inference on CPU, GPU, TPU but also to implement
the samplers in pure python/numpy code.


## Syntax

### Define random variables

The constraints on the syntax are simple: while we don't want to use the
assignment `=` operator to assign a random object, we also need for the model's
definition to be syntactically valid Python code. We choose the matrix
multiplication `@` operator. Defining a normally-disitributed random variable
thus looks like

```python
x @ Normal(0, 1)
```


### Models are generative

Bayesian models are often called *generative models*, since they can generate
random outcomes. We follow this philosophy, thus models are defined in functions
that take some input data, and returns an outcome. Take a simple 1d linear
regressions, where we would like to model the relationship between an outcome
`y` and an observation `x`:

```python
def linear_regression(x): # can be vmapped
    weight @ Normal(0, 1)
    sigma @ HalfNormal(0, 1)
    y @ Normal(x*w, sigma)
    return y
```

Or the number of successes in `N` coin tosses:

```python
def coin_tosses(N):
    p @ Beta(.5, .5)
    successes @ Binomial(p, N)
    return successes
```

The idea being:

- If I execute `linear_regression` with a coordinate `x` it should return a
  value drawn from the model conditioned on that value.
- If I execute `coin_tosses` with a number of draws, it should return a number
  of successes conditioned on this value.

When the function is executed before the distribution of the parameters are
learned from data, we call this *prior predictive sampling*. When it is executed
after sampling, *posterior predictive sampling*.

*Only the variables that are passed as arguments of the function and those
returned can be observed.*


### Models are distributions

Each model has a `sample` and a `logpdf` method, and can thus be treated as 
a distribution. This implies that once can separate a model into modules that
allow code re-use. This is valid `mcmx` code:

```python
from mymodule import my_other_prior

def my_prior(sigma):
  value @ Normal(0, sigma)
  return value

def linear_regression(x):
  weight @ my_prior(1)
  sigma @ my_other_prior(0)
  y @ Normal(x*w, sigma)
  return y
```


### Models can call any function in scope

Which allows to define complex models in a modular way. This is valid `mcmx`
code:

```python
multiply = lambda x, y: x*y

def linear_regression(x):
  weight @ Normal(1)
  sigma @ HalfNormal(0, 1)
  z = multiply(x, w)
  y @ Normal(z, sigma)
  return y
```

or if the function is defined in another module `utils`:

```python
from utils import multiply

def linear_regression(x):
  weight @ Normal(1)
  sigma @ HalfNormal(0, 1)
  z = multiply(x, w)
  y @ Normal(z, sigma)
  return y
```


### Bayesian Neural Networks are distributions over functions

*the API for random layers is not yet fixed. Many details need to be ironed.*
The following will be valid `mcmx` code. The execution of the generative model
is as follows:

1. Generate a function called `nn` by getting a sample of its weights' layers;
2. Get the category probabilities by passing the image through the sampled
   function.
3. Generate a category at random from th categorical distibution with
   the previously computed value of the vector `p`.

```python
def mnist_classifier(image):
    nn @ mcmx.layers.Serial(
        dense(400, Normal(0, 1)),
        dense(400, Normal(0, 1)),
        dense(10, Normal(0, 1)),
        softmax()
    )
    p = nn(image)
    cat @ Categorical(p)
    return cat
```

### Stochastic processes are distributions over distributions

*This is more wishful thinking at the moment than a real API proposal.*

```python
def dirichlet_process(x):
  P @ DP(alpha, P0) # P = probability distirbution
  theta @ P # theta distributed according to P, event shape can change as more
            # data appears.
  x @ Normal(theta, 1)
```
