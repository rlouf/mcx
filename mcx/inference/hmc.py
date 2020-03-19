import jax
from jax import numpy as np
import numpy

import mcx
from mcx.inference.integrators import velocity_verlet, hmc_proposal
from mcx.inference.kernels import hmc_kernel, HMCState
from mcx.inference.metrics import gaussian_euclidean_metric
from mcx.inference.runtime import Runtime


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

        position = np.stack(
            [samples[k][:num_chains] for k in self.model.posterior_variables]
        ).T
        logprob = np.array([logpdf(p) for p in position])
        logprob_grad = np.array([jax.grad(logpdf)(p) for p in position])
        initial_state = HMCState(position, logprob, logprob_grad)

        return initial_state

    def inference_kernel(self, logpdf, inverse_mass_matrix, mass_matrix_sqrt):
        """Returns the HMC runtime's transition kernel.
        """
        momentum_generator, kinetic_energy = gaussian_euclidean_metric(
            inverse_mass_matrix, mass_matrix_sqrt, self.unravel_fn,
        )
        integrator_step = velocity_verlet(logpdf, kinetic_energy)
        proposal = hmc_proposal(integrator_step, self.step_size, self.path_length)
        transition_kernel = hmc_kernel(
            proposal, momentum_generator, kinetic_energy, logpdf
        )

        def kernel(rng_key, state):
            new_state, new_info = transition_kernel(rng_key, state)
            return new_state

        return kernel

    def to_trace(self, states):
        """To build the trace we need the unraveling function to transform the
        positions back to the original shape.

        self.unravel_fn
        """

        trace = {}
        for arg, state in zip(self.model.posterior_variables, states):
            samples = state.position
            trace[arg] = numpy.asarray(samples).T.squeeze()
        return trace
