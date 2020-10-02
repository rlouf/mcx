<h2 align="center">
  /ˈmɪks/   
</h2>
   
<h3 align="center">
 XLA-rated Bayesian inference
</h3>

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

See the [documentation](https://rlouf.github.io/mcx) for more information.  See [this issue](https://github.com/rlouf/mcx/issues/1) for an updated roadmap for v0.1.

## Current API

Note that there are still many moving pieces in `mcx` and the API may change
slightly.

```python
from jax import numpy as np
import mcx
import mcx.distributions as dist

rng_key = jax.random.PRNGKey(0)
observations = {'x': x_data, 'predictions': y_data, 'lmbda': 3.}

@mcx.model
def linear_regression(x, lmbda=1.):
    scale <~ dist.Exponential(lmbda)
    coefs <~ dist.Normal(np.zeros(np.shape(x)[-1]))
    y = np.dot(x, coefs)
    predictions <~ dist.Normal(y, scale)
    return predictions

kernel = mcx.HMC(100)
sampler = mcx.sampler(
    rng_key,
    linear_regression,
    kernel,
    **observations
)
posterior = sampler.run()
```

## MCX's future

MCX's core is very flexible, so we can start considering the following
applications:

- **Neural network layers:** You can follow discussions about the API in [this Pull Request](https://github.com/rlouf/mcx/pull/16).
- **Programs with stochastic support:** Discussion in this [Issue](https://github.com/rlouf/mcx/issues/37).
- **Tools for causal inference:** Made easier by the internal representation as a
  graph.

You are more than welcome to contribute to these discussions, or suggest
potential future directions.


## Linear sampling

Like most PPL, MCX implements a batch sampling runtime:

```python
sampler = mcx.sampler(
    rng_key,
    linear_regression,
    kernel,
    **observations
)

posterior = sampler.run()
```

The warmup trace is discarded by default but you can obtain it by running:

```python
warmup_posterior = sampler.warmup()
posterior = sampler.run()
```

You can extract more samples from the chain after a run and combine the
two traces:

```python
posterior += sampler.run()
```

By default MCX will sample using a python `for` loop and display a progress bar.
For faster sampling (but without progress bar) you can use:

```python
posterior = sampler.run(accelerate=True)
```

One could use the combination in a notebook to first get a lower bound on the
sampling rate before deciding on a number of samples.


## Iterative sampling

Sampling the posterior is an iterative process. Yet most libraries only provide
batch sampling. The generator runtime is already implemented in `mcx`, which
opens many possibilities such as:

- Dynamical interruption of inference (say after getting a set number of
  effective samples);
- Real-time monitoring of inference with something like tensorboard;
- Easier debugging.

```python
samples = mcx.iterative_sampler(
    rng_key,
    linear_regression,
    kernel,
    **observations
)

for sample in samples:
  print(sample)
```

### Note

`sampler` and `iterative_sampler` share a very similar API and philosophy, they
will likely be merged before the 0.1 release:

```python
sampler = mcx.sampler(
    rng_key,
    linear_regression,
    kernel,
    **observations
)

posterior = sampler.run()

for sample in sampler:
  print(sample)
```

so it is possible to switch between the two execution modes seemlessly.

## Sequential sampling 

One of Bayesian statistics' promises is the ability to update one's knowledge as
more data becomes available. In practice, few libraries allow this in a
straightforward way. This is however important in at least two areas of
application:

- Training models with data that does not fit in memory. For deep models,
  obviously, but not necessarily;
- Training models where data progressively arrives. Think A/B testing for
  instance, where we need to update our knowledge as more users arive.
  
Sequential Markov Chain Monte-Carlo is already implemented in `mcx`. However,
more work is needed to diagnose the obtained samples and possibly stop sampling
dynamically.

```python
sampler = mcx.sequential_sampler(
    rng_key,
    linear_regression,
    kernel,
    **observations
)
posterior = sampler.run()

updated_posterior = sampler.update(posterior, **new_observations)
```

## Important note

MCX is a building atop the excellent ideas that have come up in the past 10
years of probablistic programming, whether from Stan (NUTS and the very
knowledgeable community), PyMC3 & PyMC4 (for its simple API), Tensorflow
Probability (for its shape system and inference vectorization), (Num)Pyro (for
the use of JAX in the backend), Gen.jl and Turing.jl (for composable inference),
Soss.jl (generative model API), Anglican, and many that I forget.
