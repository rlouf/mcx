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
