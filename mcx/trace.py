from dataclasses import asdict, dataclass, replace

from typing import Callable, Dict, Optional
from arviz.data.base import dict_to_dataset
from arviz import InferenceData
import jax
import jax.numpy as np

import mcx

__all__ = ["Trace"]


@dataclass
class MCXTrace:
    samples: Optional[Dict] = None
    sampling_info: Optional[Dict] = None
    loglikelihoods: Optional[Dict] = None
    warmup_samples: Optional[Dict] = None
    warmup_sampling_info: Optional[Dict] = None
    warmup_info: Optional[Dict] = None


class Trace(InferenceData):
    """Trace contains the data generated during inference: samples,
    divergences, values of diagnostics, etc.

    The class is a thin wrapper around ArviZ's InferenceData, it is an
    interface between the chains produced by the samplers and ArviZ.

    +---------+            +------------+              +------------+
    | Sampler | ---------> |  Evaluator | -----------> |   Trace    |
    +---------+  states    +------------+   samples    +------------+
                 info                       info
                 ravel_fn

    This design ensures that neither of the sampler of the trace need be aware
    of what evaluator produced the trace. This avoids carrying branching logic
    to accomodate fundamentally different algorithms (such as HMC and
    Metropolis Hastings).

    Attributes
    ----------
    state:
        The chain state as provided by the sampling algorithm.
    info:
        The chain info as provided by the sampling algorithm.
    """

    def __init__(
        self,
        *,
        samples: Dict = None,
        sampling_info: Dict = None,
        warmup_samples: Dict = None,
        warmup_sampling_info: Dict = None,
        warmup_info: Dict = None,
        loglikelihood_contributions_fn: Callable = None,
    ):
        """Build a Trace object from MCX data.

        Parameters
        ----------
        samples
            Posterior samples from a model. The dictionary maps the variables
            names to their posterior samples with shape (n_chains, num_samples, var_shape).

        """
        self._groups = []
        self._groups_warmup = []

        if samples is not None:
            pass
        elif warmup_samples is not None:
            pass
        else:
            raise ValueError(
                "To build a Trace you need at least one of samples or warmup_samples."
                " It is not recommended to build a Trace object yourself. Please raise"
                " an issue if your use case necessitates to instantiate a Trace yourself."
            )

        self.mcx = MCXTrace(
            samples=samples,
            sampling_info=sampling_info,
            warmup_samples=warmup_samples,
            warmup_sampling_info=warmup_sampling_info,
            warmup_info=warmup_info,
        )
        self.loglikelihood_contributions_fn = loglikelihood_contributions_fn

    # The following properties constitute the interface with ArviX; when called
    # by and ArviZ function they return the data in a format it can understand.
    # We prefer to keep MCX's and ArviZ's format decoupled for now, as MCX's needs
    # are currently not properly defined.

    @property
    def posterior(self):
        samples = self.mcx.samples
        return dict_to_dataset(data=samples, library=mcx)

    @property
    def warmup_posterior(self):
        samples = self.mcx.warmup_samples
        return dict_to_dataset(data=samples, library=mcx)

    @property
    def sample_stats(self):
        info = self.mcx.sampling_info
        sample_stats = {
            "lp": info["potential_energy"],
            "acceptance_probability": info["acceptance_probability"],
            "diverging": info["is_divergent"],
            "energy": info["energy"],
            "step_size": info["step_size"],
            "num_integration_steps": info["num_integration_steps"],
        }
        return dict_to_dataset(data=sample_stats, library=mcx)

    @property
    def warmup_sample_stats(self):
        info = self.mcx.warmup_sampling_info
        sample_stats = {
            "lp": info["potential_energy"],
            "acceptance_probability": info["acceptance_probability"],
            "diverging": info["is_divergent"],
            "energy": info["energy"],
            "step_size": info["step_size"],
            "num_integration_steps": info["num_integration_steps"],
        }
        return dict_to_dataset(data=sample_stats, library=mcx)

    @property
    def log_likelihood(self):
        if self.mcx.loglikelihoods:
            loglikelihoods = self.mcx.loglikelihoods
        else:

            def compute_in(samples):
                return self.loglikelihood_contributions_fn(**samples)

            def compute(samples):
                in_axes = ({key: 0 for key in self.mcx.samples},)
                return jax.vmap(compute_in, in_axes=in_axes)(samples)

            in_axes = ({key: 0 for key in self.mcx.samples},)
            loglikelihoods = jax.vmap(compute, in_axes=in_axes)(self.mcx.samples)
            self.mcx.loglikelihoods = loglikelihoods

        return dict_to_dataset(data=loglikelihoods, library=mcx)

    # The following methods are used to concatenate two traces or add new samples
    # to a trace. This can be

    def __iadd__(self, trace):
        """Concatenate this trace with another.

        We only need to concatenate the information contained in the internal
        trace.

        Examples
        --------
        This can be used to concantenate a sampling trace with the warmup trace:

            >>> trace = sampler.warmup()
            >>> trace += sampler.run()

        Or to append more samples to a trace shall we want more:

            >>> trace = sampler.run(1_000)
            >>> trace += sampler.run(10_000)

        """
        for field, new_value in asdict(trace.mcx).items():
            current_value = getattr(self.mcx, field)
            if current_value is None and new_value is not None:
                changes = {f"{field}": new_value}
                self.mcx = replace(self.mcx, **changes)
            elif current_value is not None and new_value is not None:
                stacked_values = jax.tree.multimap(np.stack, ((current_value, new_value)))
                changes = {f"{field}": stacked_values}
                self.mcx = replace(self.mcx, **changes)

    def __add__(self, trace):

    def append(self, *, samples, sampling_info):
        """Append a trace or new elements to the current trace. This is useful
        when performing repeated inference on the same dataset, or using the
        generator runtime. Sequential inference should use different traces for
        each sequence.
        """
        pass
