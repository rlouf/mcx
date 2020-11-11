from abc import ABC, abstractmethod


class Evaluator(ABC):

    @abstractmethod
    def initial_states(self, positions, loglikelihood):
        pass

    @abstractmethod
    def transform(self, model):
        pass

    @abstractmethod
    def warmup(self, model):
        pass

    @abstractmethod
    def kernel_factory(self, loglikelihood):
        pass

    @abstractmethod
    def make_trace(self, chain, unravel_fn):
        pass

    @abstractmethod
    def make_warmup_trace(self, chain, unravel_fn):
        pass
