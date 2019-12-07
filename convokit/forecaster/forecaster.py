from convokit.model import Corpus, Conversation, Utterance
from typing import Callable, Optional
from convokit import Transformer
from .cumulativeBoW import CumulativeBoW
from .forecasterModel import ForecasterModel

class Forecaster(Transformer):
    """
    Implements basic Forecaster behavior
    """
    def __init__(self, forecaster_model: ForecasterModel = None,
                 convo_structure: str = "branched",
                 text_func=lambda utt: utt.text,
                 utt_selector_func: Callable[[Utterance], bool] = lambda utt: True,
                 convo_selector_func: Callable[[Conversation], bool] = lambda convo: True,
                 pred_feat: Optional[str] = None,
                 forecast_feat_name: str = "forecast", forecast_prob_feat_name: str = "forecast_prob",
                 skip_broken_convos: bool = False):

        assert convo_structure in ["branched", "linear"]
        self.convo_structure = convo_structure

        if forecaster_model is None:
            print("No model passed to Forecaster. Initializing default forecaster model: Cumulative Bag-of-words...")
            self.forecaster_model = CumulativeBoW(forecast_feat_name=forecast_feat_name,
                                                  forecast_prob_feat_name=forecast_prob_feat_name)
        else:
            self.forecaster_model = forecaster_model

        self.pred_feat = pred_feat
        self.text_func = text_func
        self.utt_selector_func = utt_selector_func
        self.convo_selector_func = convo_selector_func
        self.forecast_feat_name = forecast_feat_name
        self.forecast_prob_feat_name = forecast_prob_feat_name
        self.skip_broken_convos = skip_broken_convos

    def _get_context_reply_label_dict(self, corpus: Corpus, include_label=True):
        dialogs = []
        if self.convo_structure == "branched":
            for convo in corpus.iter_conversations(self.convo_selector_func):
                try:
                    for path in convo.get_root_to_leaf_paths():
                        path = [utt for utt in path if self.utt_selector_func(utt)]
                        if len(path) == 1: continue
                        dialogs.append(path)
                except ValueError as e:
                    if not self.skip_broken_convos:
                        raise e

        elif self.convo_structure == "linear":
            for convo in corpus.iter_conversations(self.convo_selector_func):
                utts = convo.get_chronological_utterance_list(selector=self.utt_selector_func)
                if len(utts) == 1: continue
                dialogs.append(utts)

        id_to_context_reply_label = dict()

        for dialog in dialogs:
            for idx in range(1, len(dialog)):
                reply = self.text_func(dialog[idx])
                label = dialog[idx].meta[self.pred_feat] if include_label else None
                reply_id = dialog[idx].id
                context = [self.text_func(utt) for utt in dialog[:idx]]
                id_to_context_reply_label[reply_id] = (context, reply, label) if include_label else (context, reply, None)

        return id_to_context_reply_label

    def fit(self, corpus: Corpus, y=None):
        """
        :param corpus:
        :param y:
        :return:
        """
        id_to_context_reply_label = self._get_context_reply_label_dict(corpus, include_label=True)
        self.forecaster_model.train(id_to_context_reply_label)

    def transform(self, corpus: Corpus) -> Corpus:
        id_to_context_reply_label = self._get_context_reply_label_dict(corpus, include_label=False)
        forecast_df = self.forecaster_model.forecast(id_to_context_reply_label)

        for utt in corpus.iter_utterances():
            if utt.id in forecast_df.index:
                utt.add_meta(self.forecast_feat_name, forecast_df.loc[utt.id][self.forecast_feat_name])
                utt.add_meta(self.forecast_prob_feat_name, forecast_df.loc[utt.id][self.forecast_prob_feat_name])
            else:
                utt.add_meta(self.forecast_feat_name, None)
                utt.add_meta(self.forecast_prob_feat_name, None)

        return corpus

    def summarize(self, corpus: Corpus):
        id_to_context_reply = self._get_context_reply_label_dict(corpus, include_label=False)
        forecast_df = self.forecaster_model.forecast(id_to_context_reply)
        return forecast_df.sort_values(self.forecast_feat_name, ascending=False)

    def get_model(self):
        """
        Get the forecaster model object
        """
        return self.forecaster_model

    def set_model(self, forecaster_model):
        """
        Set the forecaster model
        :return:
        """
        self.forecaster_model = forecaster_model
