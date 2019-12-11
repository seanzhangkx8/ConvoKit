from convokit.classifier.util import extract_feats_from_obj
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
import numpy as np
from typing import List, Callable, Union, Optional
from pandas import DataFrame
from convokit.model import Corpus, Conversation, User, Utterance
from collections import defaultdict
from random import shuffle, choice
from scipy.sparse import csr_matrix
from convokit import Transformer
import pandas as pd

class PairedPrediction(Transformer):
    def __init__(self, obj_type: str,
                 pairing_func: Callable[[Union[User, Utterance, Conversation]], str],
                 pred_feats: List[str],
                 pos_label_func: Callable[[Union[User, Utterance, Conversation]], bool],
                 neg_label_func: Callable[[Union[User, Utterance, Conversation]], bool],
                 selector: Callable[[Union[User, Utterance, Conversation]], bool] = lambda x: True,
                 clf=None, pair_id_feat_name: str = "pair_id",
                 label_feat_name: str = "label",
                 pair_orientation_feat_name: str = "pair_orientation"):


        """
        DESIGN DECISION: assume that features live in metadata
        :param pairing_func: the Corpus object characteristic to pair on,
                e.g. to pair on the first 10 characters of a well-structured id, use lambda obj: obj.id[:10]
        :param pred_feats: List of metadata features to be used in prediction. Features can either be values or a
                        dictionary of key-value pairs, but not a nested dictionary
        :param pos_label_func: The function to check if the object is a positive instance
        :param neg_label_func: The function to check if the object is a negative instance
        :param selector: optional function to filter object for
        :param clf: optional classifier to be used in the paired prediction
        :param pair_id_feat_name: metadata feature name to use in annotating object with pair id, default: "pair_id"
        :param label_feat_name: metadata feature name to use in annotating object with predicted label, default: "label"
        :param pair_orientation_feat_name: metadata feature name to use in annotating object with pair orientation, default: "pair_orientation"

        """
        assert obj_type in ["user", "utterance", "conversation"]
        self.obj_type = obj_type
        self.clf = Pipeline([("standardScaler", StandardScaler(with_mean=False)),
                             ("logreg", LogisticRegression(solver='liblinear'))]) if clf is None else clf
        self.pairing_func = pairing_func
        self.pred_feats = pred_feats
        self.pos_label_func = pos_label_func
        self.neg_label_func = neg_label_func
        self.selector = selector
        self.pair_id_feat_name = pair_id_feat_name
        self.label_feat_name = label_feat_name
        self.pair_orientation_feat_name = pair_orientation_feat_name

    def _get_pos_neg_objects(self, corpus: Corpus):
        pos_objects = []
        neg_objects = []
        for obj in corpus.iter_objs(self.obj_type, self.selector):
            if not self.selector(obj): continue
            if self.pos_label_func(obj):
                pos_objects.append(obj)
            elif self.neg_label_func(obj):
                neg_objects.append(obj)
        return pos_objects, neg_objects

    def _pair_objs(self, pos_objects, neg_objects):
        """

        :param pos_objects:
        :param neg_objects:
        :return: dictionary indexed by the paired feature instance value,
                 with the value being a tuple (pos_obj, neg_obj)
        """
        pair_feat_to_pos_objs = defaultdict(list)
        pair_feat_to_neg_objs = defaultdict(list)

        for obj in pos_objects:
            pair_feat_to_pos_objs[self.pairing_func(obj)].append(obj)

        for obj in neg_objects:
            pair_feat_to_neg_objs[self.pairing_func(obj)].append(obj)

        valid_pairs = set(pair_feat_to_neg_objs).intersection(set(pair_feat_to_pos_objs))

        return {pair_id: (choice(pair_feat_to_pos_objs[pair_id]),
                             choice(pair_feat_to_neg_objs[pair_id]))
                            for pair_id in valid_pairs}


    def _generate_paired_X_y(self, pair_id_to_objs):
        pos_obj_dict = dict()
        neg_obj_dict = dict()
        for pair_id, (pos_obj, neg_obj) in pair_id_to_objs.items():
            pos_obj_dict[pair_id] = extract_feats_from_obj(pos_obj, self.pred_feats)
            neg_obj_dict[pair_id] = extract_feats_from_obj(neg_obj, self.pred_feats)
        pos_obj_df = DataFrame.from_dict(pos_obj_dict, orient='index')
        neg_obj_df = DataFrame.from_dict(neg_obj_dict, orient='index')

        X, y = [], []
        for pair_id in pair_id_to_objs:
            pos_feats = np.array(pos_obj_df.loc[pair_id])
            neg_feats = np.array(neg_obj_df.loc[pair_id])
            orientation = pair_id_to_objs[pair_id][0].meta[self.pair_orientation_feat_name]

            assert orientation in ["pos", "neg"]

            if orientation == "pos":
                y.append(1)
                diff = pos_feats - neg_feats
            else:
                y.append(0)
                diff = neg_feats - pos_feats

            X.append(diff)

        return csr_matrix(np.array(X)), np.array(y)


    def _assign_pair_orientations(self, obj_pairs):
        pair_ids = list(obj_pairs)
        shuffle(pair_ids)
        pair_orientations = dict()
        flip = True
        for pair_id in pair_ids:
            pair_orientations[pair_id] = "pos" if flip else "neg"
            flip = not flip
        return pair_orientations

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Annotate corpus objects with pair information (label, pair_id, pair_orientation)
        :param corpus:
        :return:
        """
        pos_objs, neg_objs = self._get_pos_neg_objects(corpus)
        obj_pairs = self._pair_objs(pos_objs, neg_objs)
        pair_orientations = self._assign_pair_orientations(obj_pairs)

        for pair_id, (pos_obj, neg_obj) in obj_pairs.items():
            pos_obj.add_meta(self.label_feat_name, "pos")
            neg_obj.add_meta(self.label_feat_name, "neg")
            pos_obj.add_meta(self.pair_id_feat_name, pair_id)
            neg_obj.add_meta(self.pair_id_feat_name, pair_id)
            pos_obj.add_meta(self.pair_orientation_feat_name, pair_orientations[pair_id])
            neg_obj.add_meta(self.pair_orientation_feat_name, pair_orientations[pair_id])

        for obj in corpus.iter_objs(self.obj_type):
            # unlabelled objects include both objects that did not pass the selector
            # and objects that were not selected in the pairing step
            if self.label_feat_name not in obj.meta:
                obj.add_meta(self.label_feat_name, None)
                obj.add_meta(self.pair_id_feat_name, None)
                obj.add_meta(self.pair_orientation_feat_name, None)

        return corpus

    # def fit(self, corpus: Corpus):
    #     pos_convos, neg_convos = self._get_pos_neg_objects(corpus)
    #     convo_pairs = self._pair_objs(pos_convos, neg_convos)
    #     X, y = self._generate_paired_X_y(convo_pairs)
    #     self.clf.fit(X, y)
    #     return self

    def summarize(self, corpus: Corpus, cv=LeaveOneOut()):
        """
        Run
        :param corpus:
        :param cv:
        :return:
        """
        # Check if transform() needs to be run first
        sample_obj = next(corpus.iter_objs(self.obj_type))
        meta_keys = set(sample_obj.meta)
        required_keys = {self.pair_orientation_feat_name, self.pair_id_feat_name, self.label_feat_name}
        required_keys -= meta_keys
        if len(required_keys) > 0:
            raise ValueError("Some metadata features required for paired prediction are missing: {}. "
                             "You may need to run transform() first.".format(required_keys))

        pair_id_to_obj = {'pos': dict(), 'neg': dict()}
        for obj in corpus.iter_objs(self.obj_type, self.selector):
            if obj.meta[self.pair_orientation_feat_name] is None: continue
            pair_id_to_obj[obj.meta[self.label_feat_name]][obj.meta[self.pair_id_feat_name]] = obj

        pair_ids = set(pair_id_to_obj['pos'].keys()).intersection(set(pair_id_to_obj['neg'].keys()))

        # print(set(pair_id_to_obj['pos'].keys()))
        print("Found {} valid pairs.".format(len(pair_ids)))
        pair_id_to_objs = dict()
        for pair_id in pair_ids:
            pair_id_to_objs[pair_id] = (pair_id_to_obj['pos'][pair_id], pair_id_to_obj['neg'][pair_id])

        X, y = self._generate_paired_X_y(pair_id_to_objs)
        self.clf.fit(X, y)
        return np.mean(cross_val_score(self.clf, X, y, cv=cv, error_score='raise'))

    def get_coefs(self, feature_names: List[str]):
        """
        Get dataframe of classifier coefficients, assuming it is a pipeline with a logistic regression component
        :param feature_names: list of feature names to get coefficients for
        :return: DataFrame of features and coefficients, indexed by feature names
        """
        coefs = self.clf.named_steps['logreg'].coef_[0].tolist()
        assert len(feature_names) == len(coefs)
        feats_coefs = sorted(list(zip(feature_names, coefs)), key=lambda x: x[1], reverse=True)
        return pd.DataFrame(feats_coefs, columns=['feat_name', 'coef']).set_index('feat_name')


    def print_extreme_coefs(self, feature_names: List[str], num_features: Optional[int] = None):
        """
        Must be run after summarize()
        Prints the extreme coefficients of the trained classifier model for visual inspection, assuming
        it is a pipeline with a logistic regression component
        :param feature_names: list of feature names to inspect
        :param num_features: optional number of extreme coefficients to print
        :return: None (prints features)
        """
        coefs = self.clf.named_steps['logreg'].coef_[0].tolist()

        assert len(feature_names) == len(coefs)

        feats_coefs = sorted(list(zip(feature_names, coefs)), key=lambda x: x[1], reverse=True)

        if num_features is None:
            num_features = len(feature_names) // 4

        print()
        print("TOP {} FEATURES".format(num_features))
        for ft, coef in feats_coefs[:num_features]:
            print("{}: {:.3f}".format(ft, coef))
        print()
        print("BOTTOM {} FEATURES".format(num_features))
        for ft, coef in feats_coefs[-num_features:]:
            print("{}: {:.3f}".format(ft, coef))
        print()


