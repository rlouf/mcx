from dataclasses import asdict, dataclass, replace

from typing import Callable, Dict, List, Optional, Tuple
from arviz.data.base import dict_to_dataset
from arviz import InferenceData
import jax
import jax.numpy as np

import mcx

__all__ = ["Trace"]


@dataclass
class RawTrace:
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
        loglikelihoods: Dict = None,
        loglikelihood_contributions_fn: Callable = None,
    ):
        """Build a Trace object from MCX data.

        Parameters
        ----------
        samples
            Posterior samples from a model. The dictionary maps the variables
            names to their posterior samples with shape (n_chains, num_samples, var_shape).

        """
        self._groups: List[str] = []
        self._groups_warmup: List[str] = []

        self.raw = RawTrace(
            samples=samples,
            sampling_info=sampling_info,
            loglikelihoods=loglikelihoods,
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
        samples = self.raw.samples
        return dict_to_dataset(data=samples, library=mcx)

    @property
    def warmup_posterior(self):
        samples = self.raw.warmup_samples
        return dict_to_dataset(data=samples, library=mcx)

    @property
    def sample_stats(self):
        info = self.raw.sampling_info
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
        info = self.raw.warmup_sampling_info
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
        if self.raw.loglikelihoods:
            loglikelihoods = self.raw.loglikelihoods
        else:

            def compute_in(samples):
                return self.loglikelihood_contributions_fn(**samples)

            def compute(samples):
                in_axes = ({key: 0 for key in self.raw.samples},)
                return jax.vmap(compute_in, in_axes=in_axes)(samples)

            in_axes = ({key: 0 for key in self.raw.samples},)
            loglikelihoods = jax.vmap(compute, in_axes=in_axes)(self.raw.samples)
            self.raw.loglikelihoods = loglikelihoods

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
        concatenate = lambda cur, new: np.concatenate((cur, new), axis=1)

        for field, new_value in asdict(trace.raw).items():
            current_value = getattr(self.raw, field)
            if current_value is None and new_value is not None:
                changes = {f"{field}": new_value}
                self.raw = replace(self.raw, **changes)
            elif current_value is not None and new_value is not None:
                stacked_values = jax.tree_multimap(
                    concatenate, current_value, new_value
                )
                changes = {f"{field}": stacked_values}
                self.raw = replace(self.raw, **changes)

        return self

    def __add__(self, trace):

        concatenate = lambda cur, new: np.concatenate((cur, new), axis=1)

        raw_trace_dict = {}
        for field, new_value in asdict(trace.raw).items():
            current_value = getattr(self.raw, field)
            if current_value is None and new_value is not None:
                raw_trace_dict[f"{field}"] = new_value
            elif current_value is not None and new_value is None:
                raw_trace_dict[f"{field}"] = current_value
            elif current_value is not None and new_value is not None:
                stacked_values = jax.tree_multimap(
                    concatenate, current_value, new_value
                )
                raw_trace_dict[f"{field}"] = stacked_values
            else:
                raw_trace_dict[f"{field}"] = None

        new_trace = Trace(
            **raw_trace_dict,
            loglikelihood_contributions_fn=self.loglikelihood_contributions_fn,
        )

        return new_trace

    def append(self, state: Tuple):
        """Append a trace or new elements to the current trace. This is useful
        when performing repeated inference on the same dataset, or using the
        generator runtime. Sequential inference should use different traces for
        each sequence.

        Parameter
        ---------
        state
            A tuple that contains the chain state and the corresponding sampling info.
        """
        sample, sample_info = state
        concatenate = lambda cur, new: np.concatenate((cur, new), axis=1)
        concatenate_1d = lambda cur, new: np.column_stack((cur, new))

        if self.raw.samples is None:
            stacked_chain = sample
        else:
            try:
                stacked_chain = jax.tree_multimap(concatenate, self.raw.samples, sample)
            except TypeError:
                stacked_chain = jax.tree_multimap(
                    concatenate_1d, self.raw.samples, sample
                )

        if self.raw.sampling_info is None:
            stacked_info = sample_info
        else:
            try:
                stacked_info = jax.tree_multimap(
                    concatenate, self.raw.sampling_info, sample_info
                )
            except TypeError:
                stacked_info = jax.tree_multimap(
                    concatenate_1d, self.raw.sampling_info, sample_info
                )

        self.raw = replace(self.raw, samples=stacked_chain, sampling_info=stacked_info)
