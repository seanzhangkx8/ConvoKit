from abc import ABC, abstractmethod


class ForecasterModel(ABC):

    def __init__(self, forecast_feat_name: str = "prediction", forecast_prob_feat_name: str = "score"):
        self.forecast_feat_name = forecast_feat_name
        self.forecast_prob_feat_name = forecast_prob_feat_name

    @abstractmethod
    def train(self, id_to_context_reply_label):
        pass

    @abstractmethod
    def forecast(self, id_to_context_reply_label):
        pass


