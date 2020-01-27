# MCX
   
/ˈmɪks/   

A Probabilistic Programming Library with a laser-focus
on user-friendliness, sampling and performance. Powered by JAX.

Using a Beta-Binomial model is as simple as:

```python
import mcx
import mcx.distributions as md

def beta_binomial():
    p @ md.Beta(1, 3)
    success @ md.Bernoulli(p)
    return success

model = mcx.model(beta_binomial)

print(model())
# {'b': 0.342, 'success': 1}

print(model.do(b=.5))
# {'success': 0}

# Prior predictive samples
prior_samples = model.sample(10_000)

# Inference
sampler = MCMC(model, 10_000)
trace = sampler.run()
```

MCX's philosophy

1. Knowing how to express a graphical model and manipulating Numpy arrays should
   be enough to define a model.
2. Models should be modular and re-usable.
3. Inference should be performant. Sequential inference should be a first class
   citizen.

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

## Bayesian Neural Networks

MCX currently provides basic functionalities for Bayesian Neural Networks, and
is welcoming contributions. MXMC simply subclasses `trax`'s layers to turn them
into random functions.

```python
import mcx.layers as ml
import mcx.distributions as md

def mnist_classifier(images):
    nn @ ml.Serial(
        ml.dense(400, dist=Normal(0, 1)),
        ml.dense(400, dist=Normal(0, 1)),
        ml.dense(10, dist=Normal(0, 1)),
        ml.softmax(),
    )
    p = nn(images)
    cat @ md.Categorical(p)
    return cat
```

## Important note

MCX is a building atop the excellent ideas that have come up in the past 10
years of probablistic programming, whether from Stan (NUTS and the very
knowledgeable community), PyMC3 & PyMC4 (for its simple API), Tensorflow
Probability (for its shape system and inference vectorization), (Num)Pyro (for
the use of JAX in the backend), Anglican, and many that I forget.
