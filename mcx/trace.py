from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as np
from arviz import InferenceData
from arviz.data.base import dict_to_dataset

import mcx

__all__ = ["Trace"]


@dataclass
class RawTrace:
    """Internal format for inference information."""

    samples: Optional[Dict] = None
    sampling_info: Optional[Dict] = None
    loglikelihoods: Optional[Dict] = None
    warmup_samples: Optional[Dict] = None
    warmup_sampling_info: Optional[Dict] = None
    warmup_info: Optional[Dict] = None


class Trace(InferenceData):
    """Trace contains the data generated during inference.

    The class is a thin wrapper around ArviZ's `InferenceData`, an
    interface between the chains produced by the samplers and ArviZ.

    The sampler runs the inference kernel provided by the evaluator. This
    kernel produces information in that the evaluator translates into samples
    and sampling information. This information is used to initialize a Trace
    object:

    +---------+            +------------+              +------------+
    | Sampler | ---------> |  Evaluator | -----------> |   Trace    |
    +---------+  states    +------------+   samples    +------------+
                 info                       info
                 ravel_fn

    This design ensures that neither the sampler nor the Trace need to be aware
    of what evaluator was used. This avoids carrying branching logic to
    accomodate fundamentally different algorithms (such as HMC and Metropolis
    Hastings).

    Samples and informations are stored as dictionnary internally and we interface
    with ArviZ by mimicking the `InferenceData` class. The latter's attributes are
    implemented as properties which transform the internal format to xarrays. There
    are two reasons for this design choice:

    1. Having some flexibility in the trace format. We may want to extract
    information from the trace down the line, and having control over the trace
    format means we can tweak it to improve performance. The moment we use
    ArviZ's format we lose control over that.
    2. Translating the trace to an xarray "concretizes" the DeviceArray's
    values in RAM which we do not necessarily want to do right after sampling
    in case we would like to perform other operations.
    3. Name conventions are different and it would be confusing to support two
    conventions internally.

    Examples
    --------

    A trace is immediately created when sampling from a posterior distribution:

        >>> trace_warmup = sampler.warmup()
        >>> trace_sample = sampler.run()

    It is possible to concatenate traces in place:

        >>> trace = sampler.warmup()
        >>> trace += sampler.run()

    or to concatenate two traces:

        >>> trace_1 = sampler.run()
        >>> trace_2 = sampler.run()
        >>> trace = trace_1 + trace_2

    This allows, for instance, to take first a few samples to check for convergence and
    add more samples afterwards.

    The integration with ArviZ is seemless: MCX traces can be passed to ArviZ's
    diagnostics, statistics and plotting functions.

        >>> import arvix as az
        >>> az.plot_trace(trace)


    Attributes
    ----------
    raw
        The trace in MCX's internal format.
    loglikelihood_contributions_fn
        A function that allows to contribute the variables' individual
        contribution to the loglikelihood given their values.
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
        info:
            The chain info as provided by the sampling algorithm.
        loglikelihoods
            The contributions of each variable to the loglikelihood.
        warmup_samples
            The chain states provided by the sampling algorithm as part of the warmup.
        warmup_sampling_info
            The chain info as provided by the sampling algorithm as part of the warmup.
        warmup_info
            Additional information about the warmup (e.g. value of the step size at each step).

        """
        # These are currently not updated
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
    # to a trace.

    def __iadd__(self, trace: "Trace") -> "Trace":
        """Concatenate this trace inplace with another.

        Examples
        --------
        This can be used to concantenate a sampling trace with the warmup trace:

            >>> trace = sampler.warmup()
            >>> trace += sampler.run()

        Or to append more samples to a trace shall we want more:

            >>> trace = sampler.run(1_000)
            >>> trace += sampler.run(10_000)

        Parameters
        ----------
        trace
            The trace we need to concatenate to the current one.

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

    def __add__(self, trace: "Trace") -> "Trace":
        """Concatenate the current trace with another.

        Examples
        --------
        This can be used to concatenate warmup and sampling traces or two
        sampling traces:

            >>> trace_1 = sampler.run(1_000)
            >>> trace_2 = sampler.run(10_000)
            >>> trace = trace_1 + trace_2

        Parameters
        ----------
        trace
            The trace we need to concatenate to the current one.

        Returns
        -------
        A new Trace object that contains the information contained in both the current
        trace and the trace passed as argument.

        """
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
            loglikelihood_contributions_fn=self.loglikelihood_contributions_fn,
            **raw_trace_dict,
        )

        return new_trace

    def append(self, state: Tuple) -> None:
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
