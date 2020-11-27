<h2 align="center">
  /ˈmɪks/   
</h2>

<h3 align="center">
 XLA-rated Bayesian inference
</h3>

MCX is a probabilistic programming library with a laser-focus on sampling
methods. MCX transforms the model definitions to generate logpdf or sampling
functions. These functions are JIT-compiled with JAX; they support batching and
can be exectuted on CPU, GPU or TPU transparently.

The project is currently at its infancy and a moonshot towards providing
sequential inference as a first-class citizen, and performant sampling methods
for Bayesian deep learning.

MCX's philosophy

1. Knowing how to express a graphical model and manipulating Numpy arrays should
   be enough to define a model.
2. Models should be modular and re-usable.
3. Inference should be performant and should leverage GPUs.

See the [documentation](https://rlouf.github.io/mcx) for more information.  See [this issue](https://github.com/rlouf/mcx/issues/1) for an updated roadmap for v0.1.

## Current API

Note that there are still many moving pieces in `mcx` and the API may change
slightly.

```python
import jax
import jax.numpy as np
import mcx
import mcx.distributions as dist

rng_key = jax.random.PRNGKey(0)
observations = {'x': x_data, 'predictions': y_data}

@mcx.model
def linear_regression(x, lmbda):
    sigma <~ dist.Exponential(lmbda)
    coeffs_init = np.ones(x.shape[-1])
    coeffs <~ dist.Normal(coeffs_init, sigma)
    y = np.dot(x, coeffs)
    predictions <~ dist.Normal(y, sigma)
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
posterior = sampler.run(compile=True)
```

One could use the combination in a notebook to first get a lower bound on the
sampling rate before deciding on a number of samples.


### Interactive sampling

Sampling the posterior is an iterative process. Yet most libraries only provide
batch sampling. The generator runtime is already implemented in `mcx`, which
opens many possibilities such as:

- Dynamical interruption of inference (say after getting a set number of
  effective samples);
- Real-time monitoring of inference with something like tensorboard;
- Easier debugging.

```python
samples = mcx.sampler(
    rng_key,
    linear_regression,
    kernel,
    **observations
)

trace = mcx.Trace()
for sample in samples:
  trace.append(sample)

iter(sampler)
next(sampler)
```

Note that the performance of the interactive mode is significantly lower than
that of the batch sampler. However, both can be used successively:

```python
trace = mcx.Trace()
for i, sample in enumerate(samples):
  print(do_something(sample)
  trace.append(sample)
  if i % 10 == 0:
    trace += sampler.run(100_000, compile=True)
```

## Important note

MCX takes a lot of inspiration from other probabilistic programming languages
and libraries: Stan (NUTS and the very knowledgeable community), PyMC3 (for its
simple API), Tensorflow Probability (for its shape system and inference
vectorization), (Num)Pyro (for the use of JAX in the backend), Gen.jl and
Turing.jl (for composable inference), Soss.jl (generative model API), Anglican,
and many that I forget.
