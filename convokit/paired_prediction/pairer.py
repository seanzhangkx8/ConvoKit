from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from typing import List, Callable
from convokit import Transformer, CorpusObject, Corpus
from .util import *
from convokit.classifier.util import get_coefs_helper

class Pairer(Transformer):
    def __init__(self, obj_type: str,
                 pairing_func: Callable[[CorpusObject], str],
                 pos_label_func: Callable[[CorpusObject], bool],
                 neg_label_func: Callable[[CorpusObject], bool],
                 selector: Callable[[CorpusObject], bool] = lambda x: True,
                 clf=None, pair_id_feat_name: str = "pair_id",
                 label_feat_name: str = "label",
                 pair_orientation_feat_name: str = "pair_orientation"):

        """
        :param pairing_func: the Corpus object characteristic to pair on,
                e.g. to pair on the first 10 characters of a well-structured id, use lambda obj: obj.id[:10]
        :param pos_label_func: The function to check if the object is a positive instance
        :param neg_label_func: The function to check if the object is a negative instance
        :param selector: optional function to filter object for
        :param clf: optional classifier to be used in the paired prediction
        :param pair_id_feat_name: metadata feature name to use in annotating object with pair id, default: "pair_id"
        :param label_feat_name: metadata feature name to use in annotating object with predicted label, default: "label"
        :param pair_orientation_feat_name: metadata feature name to use in annotating object with pair orientation,
        default: "pair_orientation"

        """
        assert obj_type in ["user", "utterance", "conversation"]
        self.obj_type = obj_type
        self.clf = Pipeline([("standardScaler", StandardScaler(with_mean=False)),
                             ("logreg", LogisticRegression(solver='liblinear'))]) if clf is None else clf
        self.pairing_func = pairing_func
        self.pos_label_func = pos_label_func
        self.neg_label_func = neg_label_func
        self.selector = selector
        self.pair_id_feat_name = pair_id_feat_name
        self.label_feat_name = label_feat_name
        self.pair_orientation_feat_name = pair_orientation_feat_name

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Annotate corpus objects with pair information (label, pair_id, pair_orientation)
        :param corpus: target Corpus
        :return: annotated Corpus
        """
        pos_objs, neg_objs = get_pos_neg_objects(self.obj_type, self.selector,
                                                 self.pos_label_func, self.neg_label_func, corpus)
        obj_pairs = pair_objs(self.pairing_func, pos_objs, neg_objs)
        pair_orientations = assign_pair_orientations(obj_pairs)

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