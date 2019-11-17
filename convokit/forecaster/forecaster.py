import numpy as np
from sklearn.model_selection import train_test_split
from convokit.model import Corpus, Conversation, User, Utterance
from sklearn import svm
from typing import List, Hashable, Callable, Union, Optional
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
                 text_feat: str = None,
                 utt_selector_func: Optional[Callable[[Utterance], bool]] = lambda utt: True,
                 convo_selector_func: Optional[Callable[[Conversation], bool]] = lambda convo: True):

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
        self.use_tokens = use_tokens
        self.num_tokens = num_tokens
        self.utt_selector_func = utt_selector_func
        self.convo_selector_func = convo_selector_func


    def _get_pairs(self, convo_selector_func=None, utt_exclude=None):

        pass

    def fit(self, corpus, y=None):
        """
        Fit_transform on corpus using self.vectorizer then train a classifier based on it
        :param corpus:
        :param y:
        :return:
        """
        docs = []
        for convo in corpus.iter_conversations():
            if not self.convo_selector_func(convo): continue
            for utt in convo.iter_utterances():
                if not self.utt_selector_func(utt): continue

                if self.text_feat is None:
                    docs.append(utt.text)
                else: # use text stored in metadata feature
                    utt_text = utt.get_info(self.text_feat)
                    if self.use_tokens:
                        docs.append(utt_text[:self.num_tokens])
def loadPairs(voc, corpus, split=None):
    pairs = []
    for convo in corpus.iter_conversations():
        # consider only conversations in the specified split of the data
        # if split is None or convo.meta['split'] == split:
        dialog = processDialog(voc, convo)
        for idx in range(1, len(dialog)):
            reply = dialog[idx]["tokens"][:(MAX_LENGTH-1)]
            # label = dialog[idx]["is_attack"]
            comment_id = dialog[idx]["id"]
            # gather as context all utterances preceding the reply
            context = [u["tokens"][:(MAX_LENGTH-1)] for u in dialog[:idx]]
            pairs.append((context, reply, comment_id))
            # pairs.append((context, reply, label, comment_id))
    return pairs


    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_vectorizer(self):
        return self.vectorizer

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer