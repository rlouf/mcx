from typing import Dict
import mcx
from arviz import InferenceData
from arviz.data.base import dict_to_dataset


__all__ = ["Trace"]


class Trace(InferenceData):
    """Trace contains the data generated during inference: samples,
    divergences, values of diagnostics, etc.

    The class is a thin wrapper around ArviZ's InferenceData, it is an
    interface between the chains produced by the samplers and ArviZ.

    +---------+        +------------+        +------------+
    | Sampler | -----> |  Evaluator | -----> |   Trace    |
    +---------+        +------------+        +------------+

    We do not want to have evaluator-specific logic either in the samplers
    or the trace.

    Attributes
    ----------
    state:
        The chain state as provided by the sampling algorithm.
    info:
        The chain info as provided by the sampling algorithm.
    """

    def __init__(self, *, samples: Dict = None, sampling_info: Dict = None):
        """Build a Trace object from MCX data.

        Note
        ----
        I performed an elementary benchmark where I go from returning the raw
        chains + information to an `InferenceData` object with the posterior
        and sample statistics. I found that there is no substantial difference
        in terms of performance. Therefore there is not reason to not
        initialize the `InferenceData` with both the posterior and the sample
        stats.

        Parameters
        ----------
        samples
            Posterior samples from a model. The dictionary maps the variables
            names to their posterior samples with shape (n_chains, num_samples, var_shape).
        """

        samples_dataset = dict_to_dataset(data=samples, library=mcx)

        # This will do as long as we only have samplers in the HMC family but
        # we will need to use a conversion dictionary otherwise to not
        # have specialized code here.
        sample_stats_dict = {
            "lp": sampling_info["potential_energy"],
            "acceptance_probability": sampling_info["acceptance_probability"],
            "diverging": sampling_info["is_divergent"],
            "energy": sampling_info["energy"],
            "step_size": sampling_info["step_size"],
            "num_integration_steps": sampling_info["num_integration_steps"],
        }
        samples_stats_dataset = dict_to_dataset(data=sample_stats_dict, library=mcx)

        super().__init__(posterior=samples_dataset, sample_stats=samples_stats_dataset)

    def append(self, data):
        """Append a trace or new elements to the current trace. This is useful
        when performing repeated inference on the same dataset, or using the
        generator runtime. Sequential inference should use different traces for
        each sequence.
        """
        pass
