from arviz import InferenceData


__all__ = ["Trace"]


class Trace(InferenceData):
    """Trace contains the data generated during inference: samples,
    divergences, values of diagnostics, etc.

    The class is a thin wrapper around ArviZ's InferenceData, it is an
    interface between the chains produced by the samplers and ArviZ.
    """

    def __init__(self, prior=None, posterior=None, posterior_predictive=None):
        """Extract the informations that we want to save to the trace and initializes.
        """
        thing = None
        self.posterior = None
        self.posterior_predictive = None
        self.sample_stats = None
        self.log_likelihood = None
        self.posterior_predictive = None
        super().__init__(thing)
        pass

    def to_dict(self) -> dict:
        """Returns the trace in a dictionary format that ArviZ can understand.
        The resulting dict can be persisted on disk in the json and yaml format
        to be loaded later.
        """
        return {
            "posterior": self.posterior,
            "posterior_predictive": self.posterior_predictive,
            "sample_stats": self.sample_stats,
            "log_likelihood": self.log_likelihood,
        }

    def append(self, data):
        """Append a trace or new elements to the current trace. This is useful
        when performing repeated inference on the same dataset, or using the
        generator runtime. Sequential inference should use different traces for
        each sequence.
        """
        pass
