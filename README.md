<h2 align="center">
  /ˈmɪks/   
</h2>
   
<h3 align="center">
 XLA-rated Bayesian inference
</h3>

<p align="center">
  <a href="https://github.com/rlouf/mcx/actions?query=workflow%3Abuild"><img src="https://github.com/rlouf/mcx/workflows/build/badge.svg?branch=master"></a>
  <a href="https://github.com/rlouf/mcx/actions?query=workflow%3Alint"><img src="https://github.com/rlouf/mcx/workflows/lint/badge.svg?branch=master"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


MCX is a probabilistic programing library with a laser-focus on sampling
methods. MCX transforms the model definitions to generate logpdf or sampling
functions. These functions are JIT-compiled with JAX; they support batching and
can be exectuted on CPU, GPU or TPU transparently.

The project is currently at its infancy and a moonshot towards providing
sequential inference as a first-class citizen, and performant sampling methods
for bayesian deep learning.

MCX's philosophy

1. Knowing how to express a graphical model and manipulating Numpy arrays should
   be enough to define a model.
2. Models should be modular and re-usable.
3. Inference should be performant. Sequential inference should be a first class
   citizen.

See the [documentation](https://rlouf.github.io/mcx) for more information.

## Current API

Note that there are still many moving pieces in `mcx` and the API may change
slightly. In particular, the choice of `<~` for random variable assignement may change. This is valid `mcx` code:

```python
from jax import numpy as np
import mcx
import mcx.distributions as dist

x_data = np.array([2.3, 8.2, 1.8])
y_data = np.array([1.7, 7., 3.1])

@mcx.model
def linear_regression(x, lmbda=1.):
    scale <~ dist.Exponential(lmbda)
    coefs <~ dist.Normal(np.zeros(np.shape(x)[-1]))
    y = np.dot(x, coefs)
    predictions <~ dist.Normal(y, scale)
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
kernel = mcx.HMC(num_integration_steps=100)

observations = {'x': x_data, 'predictions': y_data, 'lmbda': 3.}
sampler = mcx.sample(
    rng_key,
    linear_regression,
    kernel,
    **observations
)
trace = sampler.run()
```

## Currently implemented

* Parsing simple model definitions and compilation to `logpdf` or prior sampler;
* Sampling from the model's prior definition, prior predictive sampling;
* Bernoulli, Beta, Binomial, Categorical, Discrete Uniform, Log-Normal, Normal,
  Poisson, Uniform distributions;
* Sampling with Hamiltonian Monte Carlo;
* Batch, iterative and sequential sampling runtimes;
* core warmup logic for HMC and empirical HMC;
* Random Walk Metropolis kernel;
* 4 symplectic integrators (velocity Verlet, McLachlan, Yoshida, Four stages)

See [this issue](https://github.com/rlouf/mcx/issues/1) for an updated roadmap for v0.1.

You can follow discussions about the API for neural network layers in [this Pull Request](https://github.com/rlouf/mcx/pull/16). You are welcome to contribute to the discussion.

## Iterative sampling

Sampling the posterior is an iterative process. Yet most libraries only provide batch sampling. The generator runtime is already implemented in `mcx`, which opens many possibilities such as:

- Dynamical interruption of inference (say after getting a set number of effective samples);
- Real-time monitoring of inference with something like tensorboard;

```python
samples = mcx.generate(
    rng_key,
    linear_regression,
    kernel,
    **observations
)

for sample in samples:
  print(sample)
```

## Sequential Markov Chain Monte Carlo

One of Bayesian statistics' promises is the ability to update one's knowledge as
more data becomes available. In practice, few libraries allow this in a
straightforward way. This is however important in at least two areas of
application:

- Training models with data that does not fit in memory. For deep models,
  obviously, but not necessarily;
- Training models where data is not all available at a point in time, but rather
  progressively arrives. Think A/B testing for instance, where we need to update
  our knowledge as more users arive.
  
Sequential Markov Chain Monte-Carlo is already implemented in `mcx`. However, more work is needed to diagnose the obtained samples and possibly stop sampling dynamically.

```python
sampler = mcx.sequential(
    rng_key,
    linear_regression,
    kernel,
    **observations
)

trace_1 = sampler.update(**observations_1)
trace_2 = sampler.update(**observations_2)
```


## Important note

MCX is a building atop the excellent ideas that have come up in the past 10
years of probablistic programming, whether from Stan (NUTS and the very
knowledgeable community), PyMC3 & PyMC4 (for its simple API), Tensorflow
Probability (for its shape system and inference vectorization), (Num)Pyro (for
the use of JAX in the backend), Anglican, and many that I forget.
