from jax import random
import jax.numpy as np
import numpy as onp
import mcx
import mcx.distributions as dist


@mcx.model
def my_unif(x, lmbda=1.):
    """
    Simple model: the data doesn't depend on the parameter so the posterior will
    be the same as the prior
    """
    sigma <~ dist.Uniform(0,1)
    predictions <~ dist.Normal(0, 1)
    return predictions

kernel = mcx.HMC(100)


rng_key = random.PRNGKey(2)
rng_key, subkey = random.split(rng_key)

x_data = np.linspace(0, 5, 1000).reshape(1000, 1)
forward_samples = mcx.sample_forward(subkey, my_unif, x=x_data)
prior_names = ['sigma']
prior_samples = {name: forward_samples[name] for name in prior_names}
observations = {'x': x_data, 'predictions': forward_samples['predictions']}

sampler = mcx.sampler(
    rng_key,
    my_unif,
    kernel,
    num_chains=1,
    **observations
)

posterior = sampler.run(num_samples=500000, compile=True)
onp.savetxt("my_unif_samples.txt", posterior.posterior.sigma.data)
print(posterior.posterior.sigma.data.mean())
