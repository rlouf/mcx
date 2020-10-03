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
    """

    def __init__(self, *, posterior: Dict = None, coords=None, dims=None):
        """Build a Trace object from MCX data.

        Parameters
        ----------
        posterior
            Posterior samples from a model. The dictionary maps the variables
            names to their posterior samples with shape (n_chains, num_samples, var_shape).
        """
        samples_dataset = dict_to_dataset(
            data=posterior["samples"], library=mcx, coords=coords, dims=dims
        )

        sample_stats_dict = {
            "lp": posterior["potential_energy"],
            "mean_tree_accept": posterior["acceptance_probability"],
            "diverging": posterior["is_divergent"],
            "energy": posterior["energy"],
        }
        samples_stats_dataser = dict_to_dataset(
            data=sample_stats_dict, library=mcx, coords=coords, dims=dims
        )

        super().__init__(posterior=samples_dataset, sample_stats=samples_stats_dataser)

    def append(self, data):
        """Append a trace or new elements to the current trace. This is useful
        when performing repeated inference on the same dataset, or using the
        generator runtime. Sequential inference should use different traces for
        each sequence.
        """
        pass
