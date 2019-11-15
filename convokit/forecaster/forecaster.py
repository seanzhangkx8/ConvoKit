import numpy as np
from sklearn.model_selection import train_test_split
from convokit.model import Corpus, Conversation, User, Utterance
from sklearn import svm
from typing import List, Hashable, Callable, Union
from convokit import Transformer
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class Forecaster(Transformer):
    """
    Implements cumulative BoW Forecaster
    TODO think about convo / utterance selector functions very carefully
    """
    def __init__(self, pred_feat: str, vectorizer=None, model=None,
                 use_tokens: bool = False, num_tokens: int = 80,
                 text_feat: str = None):

        if vectorizer is None:
            print("Initializing default unigram CountVectorizer...")
            if use_tokens:
                if text_feat is None:
                    raise ValueError("You must specify where the tokens are stored in "
                                     "the Utterance metadata using the 'text_feat' parameter.")
                self.vectorizer = CV(decode_error='ignore', min_df=10, max_df=.5, ngram_range=(1, 1), binary=False,
                                     max_features=15000, tokenizer=lambda x: x, preprocessor=lambda x: x)
            else:
                self.vectorizer = CV(decode_error='ignore', min_df=10, max_df=.5,
                                     ngram_range=(1, 1), binary=False, max_features=15000)
        else:
            self.vectorizer = vectorizer

        if model is None:
            print("Initializing default forecaster model (standard scaled logistic regression)")
            self.model = Pipeline([("standardScaler", StandardScaler()),
                                   ("logreg", LogisticRegression(solver='liblinear'))])
        else:
            self.model = model

        self.pred_feat = pred_feat
        self.text_feat = text_feat
        self.num_tokens = num_tokens


    def _get_pairs(self, convo_selector_func=None, utt_exclude=None):

        pass

    def fit(self, corpus):
        """
        Fit_transform on corpus using self.vectorizer then train a classifier based on it
        :param corpus:
        :return:
        """
        if self.text_feat is None: # use utterance text
            pass
        else: # use text stored 
            if

        docs =
        for convo in corpus.iter_conversations():
            for utt in convo.iter_utterances():
                pass
        pass

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_vectorizer(self):
        return self.vectorizer

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer