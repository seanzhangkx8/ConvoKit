from abc import ABC, abstractmethod

class Framework(ABC):

    def fit(self, *args):
        pass

    @abstractmethod
    def predict(self, *args):

        pass

    def fit_predict(self, *args):
        pass