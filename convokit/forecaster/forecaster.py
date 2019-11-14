import numpy as np
from sklearn.model_selection import train_test_split
from convokit.model import Corpus, Conversation, User, Utterance
from sklearn import svm
from typing import List, Hashable, Callable, Union
from convokit import Transformer


class Forecaster(Transformer):
    def __init__(self, obj_type: str, pred_feats: List[Hashable],
                 y_func: Callable[[Union[User, Utterance, Conversation]], bool],
                 filter_func: Callable[[Union[User, Utterance, Conversation]], bool] = None, clf=None):
        self.pred_feats = pred_feats
        self.y_func = y_func
        self.filter_func = filter_func
        self.obj_type = obj_type
        self.clf = svm.SVC(C=0.02, kernel='linear', probability=True) if clf is None else clf

    def _get_pairs(self, convo_selector_func=None, utt_exclude=None):
        pass

    def fit(self, corpus):
        for convo in corpus.iter_conversations():
            for utt in convo.iter_utterances():
                pass
        pass