from .frame import Frame
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Union
from pandas import DataFrame
from convokit.model import Corpus, Conversation
from collections import defaultdict
from random import shuffle, choice

class PairedPrediction(Frame):
    def __init__(self, pairing_feat, pred_feats, pos_label_func, neg_label_func, filter_func=None,
                       clf=None, exclude_na=True, impute_na=None):
        """
        DESIGN DECISION: assume that features live in metadata, not data

        :param pairing_feat: the Conversation feature to pair on
        :param pred_feats: List of features to be used in prediction
        :param pos_label_func: The function to check if the Conversation is a positive instance
        :param neg_label_func: The function to check if the Conversation is a negative instance
        :param filter_func: optional function to filter Conversations for
        :param clf: optional classifier to be used in the paired prediction
        :param exclude_na: whether to exclude Conversations with NaN values as features
        :param impute_na: optional value to replace NaN values
        """
        self.clf = Pipeline([("standardScaler", StandardScaler()),
                             ("logreg", LogisticRegression(solver='liblinear'))]) if clf is None else clf
        self.pairing_feat = pairing_feat
        self.pred_feats = pred_feats
        self.pos_label_func = pos_label_func
        self.neg_label_func = neg_label_func
        self.filter_func = filter_func
        # check for consistent instructions for handling NaN values
        if (exclude_na and impute_na is not None) or (not exclude_na and impute_na is None):
            raise ValueError("exclude_na and impute_na arguments are not consistent")
        self.exclude_na = exclude_na
        self.impute_na = impute_na

    def _get_pos_neg_convos(self, corpus: Corpus):
        pos_convos = []
        neg_convos = []
        for convo in corpus.iter_conversations():
            if not self.filter_func(convo): continue
            if self.pos_label_func(convo):
                pos_convos.append(convo)
            elif self.neg_label_func(convo):
                neg_convos.append(convo)
        return pos_convos, neg_convos

    def _extract_convo_features(self, convo: Conversation):
        return {feat_name: convo.meta[feat_name] for feat_name in self.pred_feats}

    def _pair_convos(self, pos_convos, neg_convos):
        """

        :param pos_convos:
        :param neg_convos:
        :return: dictionary indexed by the paired feature instance value,
                 with the value being a tuple (pos_convo, neg_convo)
        """
        pair_feat_to_pos_convos = defaultdict(list)
        pair_feat_to_neg_convos = defaultdict(list)

        for convo in pos_convos:
            pair_feat_to_pos_convos[convo.meta[self.pairing_feat]].append(convo)

        for convo in neg_convos:
            pair_feat_to_neg_convos[convo.meta[self.pairing_feat]].append(convo)

        valid_pairs = set(list(pair_feat_to_neg_convos)).union(set(list(pair_feat_to_pos_convos)))

        return {valid_pair: (choice(pair_feat_to_pos_convos[valid_pair]),
                             choice(pair_feat_to_neg_convos[valid_pair]))
                            for valid_pair in valid_pairs}

    def _generate_paired_X_y(self, convo_pairs):
        pos_convo_dict = dict()
        neg_convo_dict = dict()
        for pair_id, (pos_convo, neg_convo) in convo_pairs.items():
            pos_convo_dict[pair_id] = self._extract_convo_features(pos_convo)
            neg_convo_dict[pair_id] = self._extract_convo_features(neg_convo)
        pos_convo_df = DataFrame.from_dict(pos_convo_dict, orient='index')
        neg_convo_df = DataFrame.from_dict(neg_convo_dict, orient='index')

        pair_ids = shuffle(list(convo_pairs))

        X, y = [], []
        flip = True
        excluded = 0
        for pair_id in pair_ids:
            pos_feats = np.array(pos_convo_df.loc[pair_id])
            neg_feats = np.array(neg_convo_df.loc[pair_id])

            if self.exclude_na and (np.isnan(pos_feats).any() or np.isnan(neg_feats).any()):
                excluded += 1
                continue

            if flip:
                y.append(1)
                diff = pos_feats - neg_feats
            else:
                y.append(0)
                diff = neg_feats - pos_feats

            X.append(diff)
            flip = not flip

        if excluded > 0:
            print("Excluded {} data point(s) that contained NaN values.".format(excluded))

        return np.array(X), np.array(y)

    def fit(self, corpus: Corpus):
        pos_convos, neg_convos = self._get_pos_neg_convos(corpus)
        convo_pairs = self._pair_convos(pos_convos, neg_convos)
        X, y = self._generate_paired_X_y(convo_pairs)
        self.clf.fit(X, y)

    def evaluate(self, corpus: Corpus):
        pos_convos, neg_convos = self._get_pos_neg_convos(corpus)
        convo_pairs = self._pair_convos(pos_convos, neg_convos)
        X, y = self._generate_paired_X_y(convo_pairs)
        try:
            preds = self.clf.predict(X)
            return np.mean(preds == y)
        except NotFittedError:
            print("Failed evaluation: fit() must be run first.")

    def fit_evaluate(self, corpus: Corpus, test_size: int = 0.2):
        pos_convos, neg_convos = self._get_pos_neg_convos(corpus)
        convo_pairs = self._pair_convos(pos_convos, neg_convos)
        X, y = self._generate_paired_X_y(convo_pairs)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        return np.mean(preds == y)

    def print_extreme_coefs(self, feature_names: List[str], num_features: int = 5):
        coefs = self.clf.named_steps['logreg'].coef_[0].tolist()

        assert len(feature_names) == len(coefs)

        feats_coefs = sorted(list(zip(feature_names, coefs)), key=lambda x: x[1], reverse=True)

        print()
        print("TOP {} FEATURES".format(num_features))
        for ft, coef in feats_coefs[:num_features]:
            print("{}: {:.3f}".format(ft, coef))
        print()
        print("BOTTOM {} FEATURES".format(num_features))
        for ft, coef in feats_coefs[-num_features:]:
            print("{}: {:.3f}".format(ft, coef))
        print()


