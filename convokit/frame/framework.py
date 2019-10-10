from abc import ABC, abstractmethod
from convokit.model import Corpus

class Framework(ABC):

    def fit(self, corpus: Corpus):
        pass

    @abstractmethod
    def evaluate(self, corpus: Corpus):

        pass

    def fit_evaluate(self, corpus: Corpus):
        pass