import numpy as np
from sklearn.model_selection import train_test_split
from convokit.model import Corpus, Conversation, User, Utterance
from sklearn import svm
from typing import List, Hashable, Callable, Union
from .util import extract_feats_and_label, extract_feats, extract_feats_from_obj
from .framework import Framework


class Classifier(Framework):
    def __init__(self, obj_type: str, pred_feats: List[Hashable],
                 y_func: Callable[[Union[User, Utterance, Conversation]], bool],
                 filter_func: Callable[[Union[User, Utterance, Conversation]], bool]=None, clf=None):
        self.pred_feats = pred_feats
        self.y_func = y_func
        self.filter_func = filter_func
        self.obj_type = obj_type
        self.clf = svm.SVC(C=0.02, kernel='linear', probability=True) if clf is None else clf

    def fit(self, corpus: Corpus):
        X, y = extract_feats_and_label(corpus, self.obj_type, self.pred_feats, self.y_func, self.filter_func)
        self.clf.fit(X, y)

    def evaluate(self, corpus: Corpus = None, objs: List[Union[User, Utterance, Conversation]] = None):
        assert (corpus is None and objs is not None) or (corpus is not None and objs is None)
        if objs is None:
            X = extract_feats(corpus, self.obj_type, self.pred_feats, self.filter_func)
        else:
            X = np.array([list(extract_feats_from_obj(obj, self.pred_feats).values()) for obj in objs])

        return self.clf.predict(X), self.clf.predict_proba(X)

    def fit_evaluate(self, corpus: Corpus, test_size: int = None):
        X, y = extract_feats_and_label(corpus, self.obj_type, self.pred_feats, self.y_func, self.filter_func)

        if test_size is None:
            self.clf.fit(X, y)
            return self.clf.predict(X), self.clf.predict_proba(X)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            self.clf.fit(X_train, y_train)
            preds = self.clf.predict(X_test)
            return np.mean(preds == y_test)
