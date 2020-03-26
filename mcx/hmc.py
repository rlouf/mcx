from typing import NamedTuple

import jax
from jax import numpy as np
from jax.flatten_util import ravel_pytree

from mcx import sample_forward
from mcx.core import compile_to_logpdf
from mcx.inference.integrators import velocity_verlet, hmc_proposal
from mcx.inference.kernels import hmc_kernel, HMCState
from mcx.inference.metrics import gaussian_euclidean_metric


class HMCParameters(NamedTuple):
    step_size: float
    num_integration_steps: float
    mass_matrix_sqrt: np.DeviceArray
    inverse_mass_matrix: np.DeviceArray


def HMC(model, step_size=None, num_integration_steps=None, mass_matrix_sqrt=None, inverse_mass_matrix=None, is_mass_matrix_diagonal=True):

    artifact = compile_to_logpdf(model.graph, model.namespace)
    logpdf = artifact.compiled_fn
    parameters = HMCParameters(step_size, num_integration_steps, mass_matrix_sqrt, inverse_mass_matrix)

    def _flatten_logpdf(logpdf, unravel_fn):
        def flattened_logpdf(array):
            kwargs = unravel_fn(array)
            return logpdf(**kwargs)
        return flattened_logpdf

    def initialize(rng_key, num_chains, **kwargs):
        """
        kwargs: a dictionary of arguments and variables we condition on and
                their value.
        """

        conditioning_vars = set(kwargs.keys())
        model_randvars = set(model.random_variables)
        model_args = set(model.arguments)
        available_vars = model_randvars.union(model_args)

        # The variables passed as an argument to the initialization (variables
        # on which the logpdf is conditionned) must be either a random variable
        # or an argument to the model definition.
        if not available_vars.issuperset(conditioning_vars):
            unknown_vars = list(conditioning_vars.difference(available_vars))
            unknown_str = ", ".join(unknown_vars)
            raise AttributeError("You passed a value for {} which are neither random variables nor arguments to the model definition.".format(unknown_str))

        # The user must provide a value for all of the model definition's
        # positional arguments.
        model_posargs = set(model.posargs)
        if model_posargs.difference(conditioning_vars):
            missing_vars = (model_posargs.difference(conditioning_vars))
            missing_str = ", ".join(missing_vars)
            raise AttributeError("You need to specify a value for the following arguments: {}".format(missing_str))

        # Condition on data to obtain the model's log-likelihood
        loglikelihood = jax.partial(logpdf, **kwargs)

        # Sample one initial position per chain from the prior
        to_sample_vars = model_randvars.difference(conditioning_vars)
        samples = sample_forward(rng_key, model, num_samples=num_chains, **kwargs)
        initial_positions = dict((var, samples[var]) for var in to_sample_vars)

        positions = []
        for i in range(num_chains):
            position = {k: value[i] for k, value in initial_positions.items()}
            flat_position, unravel_fn = ravel_pytree(position)
            positions.append(flat_position)
        positions = np.stack(positions)

        # Transform the likelihood to use a flat array as single argument
        flat_loglikelihood = _flatten_logpdf(loglikelihood, unravel_fn)

        # Compute the log probability and gradient to define initial state
        logprobs, logprob_grads = jax.vmap(jax.value_and_grad(flat_loglikelihood))(positions)
        initial_state = HMCState(positions, logprobs, logprob_grads)

        return flat_loglikelihood, initial_state, parameters, unravel_fn

    def build_kernel(logpdf, parameters):
        """Builds the kernel that moves the chain from one point
        to the next.
        """

        try:
            mass_matrix_sqrt = parameters.mass_matrix_sqrt
            inverse_mass_matrix = parameters.inverse_mass_matrix
            num_integration_steps = parameters.num_integration_steps
            step_size = parameters.step_size
        except AttributeError:
            AttributeError(
                "The Hamiltonian Monte Carlo algorithm requires the following parameters: mass matrix, inverse mass matrix and step size."
            )

        momentum_generator, kinetic_energy = gaussian_euclidean_metric(
            mass_matrix_sqrt, inverse_mass_matrix,
        )
        integrator = velocity_verlet(logpdf, kinetic_energy)
        proposal = hmc_proposal(integrator, step_size, num_integration_steps)
        kernel = hmc_kernel(proposal, momentum_generator, kinetic_energy, logpdf)

        return kernel

    def to_trace(states_chain, ravel_fn):
        """ Translate the raw chains to a format that can be understood by and
        is useful to humans.
        """
        return states_chain

    return initialize, build_kernel, to_trace
