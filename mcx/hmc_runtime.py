from typing import Callable

import jax

import mcx
from jax import numpy as np
from mcx.inference.metrics import gaussian_euclidean_metric
from mcx.inference.integrators import velocity_verlet, hmc_proposal
from mcx.inference.kernels import hmc_kernel, HMCState

import numpy


class Runtime(object):
    def __init__(self, model, state):
        self.state = state
        self.model = model
        self.rng_key = model.rng_key

    def logpdf_fn(self):
        raise NotImplementedError

    def warmup(self, initial_states, num_iterations):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def inference_kernel(self, logpdf, warmup_state):
        raise NotImplementedError

    def to_trace(self, states):
        raise NotImplementedError


class HMC(Runtime):
    """Hamiltonian Monte Carlo runtime.

    Programs carry a state with them; when they are re-sampled after a first
    iteration they carry out computations from the last sampled position.

    >>> algorithm = HMC(model)
    ... samples = mcx.sample(algorithm, x=np.array([1., 2.]), y=np.array([2, 3]))
    ... new_samples = mcx.sample(algorithm, ...)
    """

    def __init__(
        self, model, step_size, path_length, is_mass_matrix_diagonal=True, state=None
    ):
        super().__init__(model, state)
        self.is_mass_matrix_diagonal = is_mass_matrix_diagonal
        self.step_size = step_size
        self.path_length = path_length
        # If step size is not None run the warmup otherwise meh

    @property
    def logpdf_fn(self):
        artifact = mcx.core.compile_to_logpdf(self.model.graph, self.model.namespace)
        return artifact.compiled_fn

    def warmup(self, initial_states, num_iterations):
        """Return a function that implements the warmup."""
        pass

    def initialize(self, logpdf, num_warmup, num_chains, **kwargs):
        samples = mcx.sample_forward(self.model, num_samples=num_chains, **kwargs)
        position = np.stack([samples[k][:num_chains] for k in self.model.posterior_variables]).T
        logprob = np.array([logpdf(*p) for p in position])
        logprob_grad = np.array([jax.grad(logpdf)(*p) for p in position])
        initial_state = HMCState(position, logprob, logprob_grad)
        _, unravel_fn = jax.flatten_util.ravel_pytree(np.array([1., 2.]))
        self.unravel_fn = unravel_fn
        return initial_state

    def inference_kernel(self, logpdf, inverse_mass_matrix, mass_matrix_sqrt):
        """Returns the HMC runtime's transition kernel.
        """
        momentum_generator, kinetic_energy = gaussian_euclidean_metric(
            inverse_mass_matrix, mass_matrix_sqrt, self.unravel_fn
        )
        integrator_step = velocity_verlet(logpdf, kinetic_energy)
        proposal = hmc_proposal(integrator_step, self.step_size, self.path_length)
        transition_kernel = hmc_kernel(proposal, momentum_generator, kinetic_energy, logpdf)

        def kernel(rng_key, state):
            new_state, new_info = transition_kernel(rng_key, state)
            return new_state

        return kernel

    def to_trace(self, states):
        trace = {}
        for arg, state in zip(self.model.posterior_variables, states):
            samples = state.position
            trace[arg] = numpy.asarray(samples).T.squeeze()
        return trace


def sample(runtime: Runtime, num_samples=1000, num_warmup=1000, num_chains=4, **kwargs):
    """Sample is a runtime: it is in charge of executing the program and manage
    the input (data) - output.
    """

    # Check that all necessary variables were passed (it will yell anyway)
    logpdf = runtime.logpdf_fn
    logpdf = jax.partial(logpdf, **kwargs)

    initial_state = runtime.state
    if runtime.state is None:
        initial_state = runtime.initialize(logpdf, num_warmup, num_chains, **kwargs)

    # Create and compile the inference kernel
    kernel = runtime.inference_kernel(
        logpdf, np.array([1.0, 1.0]), np.array([1.0, 1.0])
    )
    kernel = jax.jit(kernel)

    # Run the inference
    @jax.jit
    def chain_update(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        new_states = jax.vmap(kernel)(keys, states)
        return new_states, new_states

    rng_keys = jax.random.split(runtime.rng_key, num_samples)
    states = jax.lax.scan(chain_update, initial_state, rng_keys)

    trace = runtime.to_trace(states)

    return trace
