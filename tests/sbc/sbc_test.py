from jax import random
import jax.numpy as np
import numpy as onp
import mcx
import mcx.distributions as dist
import time

from emcee.autocorr import integrated_time


def effective_sample_size(chain):
    num_samples = len(chain)
    tau_chain = max(integrated_time(chain, tol=100, quiet=True), 1, )
    ess = int(num_samples/tau_chain)
    return ess, int(tau_chain)

def rank_statistic(post_samples, prior_sample):
    list_bools = [sam < prior_sample for sam in post_samples]
    return sum(list_bools)


def run_sbc_test(rng_key, model, kernel, x, L, num_replicas):
    """
    Simulation Based Calibration: https://arxiv.org/abs/1804.06788

    Parameters
    ----------

    rng_key: Jax random key

    model: Mcx model

    kernel: function
        MCMC kernel
    x: ndarray
        array of x values for data.
    L: int
        Number of iid posterior samples to keep at each iteration.
        This corresponds to the number of bins in SBC.
    num_replicas: int
        Number of times to run SBC.
        This corresponds to the number of rank statistics you get.
    """
    prior_names = [elem for elem in model.graph.random_variables if elem!='predictions']
    print(f"Running SBC for for parameters {' '.join(prior_names)}")
    rank_statistic_dict = {name: [] for name in prior_names}

    starttime = time.time()
    for i in range(num_replicas):
        print(f"\nIteration {i}/{num_replicas}")
        rng_key, subkey = random.split(rng_key)

        # generate prior and data
        forward_samples = mcx.sample_forward(subkey, model, x=x)
        prior_samples = {name: forward_samples[name] for name in prior_names}
        observations = {'x': x, 'predictions': forward_samples['predictions'].reshape(len(x),1)}


        # build sampler
        sampler = mcx.sampler(
            rng_key,
            model,
            kernel,
            num_chains=1,
            **observations
        )

        # run sampler
        num_samples = 5000
        posterior = sampler.run(num_samples=num_samples, compile=True)

        posterior_samples = {name: posterior.posterior[name].data[0,:] for name in prior_names}
        ESS_tau_dict = {name: effective_sample_size(posterior_samples[name]) for name in prior_names}

        # if an IAT (tau) is smaller than number of samples, rerun the chan for longer
        is_chain_short = [tau>num_samples/100 for _, tau in ESS_tau_dict.values()]
        while any(is_chain_short):
            num_samples *= 2
            print(f"Running sampler for {num_samples} more iterations")
            posterior = sampler.run(num_samples=num_samples, compile=True)
            posterior_samples = {name: posterior.posterior[name].data[0,:] for name in prior_names}
            ESS_tau_dict = {name: effective_sample_size(posterior_samples[name]) for name in prior_names}
            is_chain_short = [tau>(num_samples/100) for _, tau in ESS_tau_dict.values()]

        # thin chain using IAT, and truncate to keep L samples if they're too longer
        posterior_samples = {k: v[::tau][:L] for (k,v), (_, tau) in zip(posterior_samples.items(), ESS_tau_dict.values())}

        for name, rs_list in rank_statistic_dict.items():
            rs_list.append(rank_statistic(posterior_samples[name], prior_samples[name]))

    endtime = time.time()
    print(f"Running time: {(endtime-starttime)/60:.1f}min")
    return rank_statistic_dict
