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

## Currently implemented

* Parsing simple model definitions and compilation to `logpdf` or prior sampler;
* Sampling from the model's prior definition, prior predictive sampling;
* Bernoulli, Beta, Binomial, Categorical, Discrete Uniform, Log-Normal, Normal,
  Poisson, Uniform distributions;
* Sampling with the Random Walk Metropolis algorithm;
* Sampling with Hamiltonian Monte Carlo;
* 4 symplectic integrators (velocity Verlet, McLachlan, Yoshida, Four stages)

## Roadmap

* Gamma, Dirichlet, Multivariate Normal distribution
* Hamiltonian Monte Carlo
* NUTS
* Metropolis within Gibbs block sampling for discrete variables
* Bayesian deep learning layers
* Sequential Markov Chain Monte Carlo
* Discrete Hamiltonian Monte Carlo

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


## Important note

MCX is a building atop the excellent ideas that have come up in the past 10
years of probablistic programming, whether from Stan (NUTS and the very
knowledgeable community), PyMC3 & PyMC4 (for its simple API), Tensorflow
Probability (for its shape system and inference vectorization), (Num)Pyro (for
the use of JAX in the backend), Anglican, and many that I forget.
