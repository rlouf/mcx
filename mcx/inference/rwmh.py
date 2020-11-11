import inspect
from typing import Callable, Tuple

import jax
import jax.numpy as np

from mcx.inference.evaluator import Evaluator
from mcx.inference.kernels import RWMState, RWMInfo, rwm_kernel


class RWMH(Evaluator):
    """Random Walk Metropolis Hastings evaluator."""
    def __init__(self, proposal_generator, **parameters):
        self.needs_warmup = True
        try:
            proposal_generator(**parameters)
            self.needs_warmup = False
        except TypeError:
            proposal_args = inspect.getfullargspec(proposal_generator)[0]
            proposal_args_str = ", ".join(proposal_args)
            raise TypeError(
                f"The proposal '{proposal_generator.__name__}' takes {proposal_args_str} "
                f"as arguments. You passed {','.join(parameters.keys())} instead. "
                "You can also choose to not pass any argument in wich case they will be "
                "automatically adapted."
            )

        self.proposal_generator = proposal_generator
        self.parameters = parameters

    def transform(self, model):
        return model

    def states(self, positions, loglikelihood):

        @jax.jit
        def make_state(position):
            log_prob = loglikelihood(position)
            return RWMState(position, log_prob)

        states = jax.vmap(make_state)(positions)

        return states

    def warmup(
        self,
        rng_key: jax.random.PRNGKey,
        initial_state: RWMState,
        kernel_factory: Callable,
        num_chains: int,
        num_warmup_steps: int = 1000,
        compile=False,
    ):
        if not self.needs_warmup:
            parameters = (np.ones(num_chains) * self.parameters['sigma'],)
            return initial_state, parameters, None

        pass

    def kernel_factory(self, loglikelihood: Callable) -> Callable:

        def build_kernel(parameter):
            proposal = self.proposal_generator(parameter)
            kernel = rwm_kernel(loglikelihood, proposal)
            return kernel

        return build_kernel

    def make_trace(self, chain: Tuple[RWMState, RWMInfo], unravel_fn: Callable):
        state, info = chain

        unravel_chain = jax.vmap(unravel_fn, in_axes=(0,))
        try:
            samples = jax.vmap(unravel_chain, in_axes=(1,))(state.position)
        except IndexError:  # single chain
            samples = unravel_chain(state.position)

        sampling_info = {
            "acceptance_probability": info.acceptance_probability.T,  # type: ignore
            "is_accepted": info.is_accepted.T,  # type: ignore
            "log_prob": state.log_prob.T,  # type: ignore
        }

        return samples, sampling_info

    def make_warmup_trace(self, chain, unravel_fn):
        pass
