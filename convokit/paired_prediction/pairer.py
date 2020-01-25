from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import Callable
from convokit import Transformer, CorpusObject, Corpus
from .util import *
from collections import defaultdict
from random import shuffle, seed

class Pairer(Transformer):
    def __init__(self, obj_type: str,
                 pairing_func: Callable[[CorpusObject], str],
                 pos_label_func: Callable[[CorpusObject], bool],
                 neg_label_func: Callable[[CorpusObject], bool],
                 pair_mode: str = "random",
                 selector: Callable[[CorpusObject], bool] = lambda x: True,
                 pair_id_feat_name: str = "pair_id",
                 label_feat_name: str = "pair_obj_label",
                 pair_orientation_feat_name: str = "pair_orientation"):

        """
        :param pairing_func: the Corpus object characteristic to pair on,
                e.g. to pair on the first 10 characters of a well-structured id, use lambda obj: obj.id[:10]
        :param pos_label_func: The function to check if the object is a positive instance
        :param neg_label_func: The function to check if the object is a negative instance
        :param pair_mode: 'random': pick a single positive and negative object pair randomly (default),
                          'maximize': pick the maximum number of positive and negative object pairs possible randomly,
                       or 'first': pick the first positive and negative object pair found.
        :param selector: optional function to filter object for
        :param clf: optional classifier to be used in the paired prediction
        :param pair_id_feat_name: metadata feature name to use in annotating object with pair id, default: "pair_id"
        :param label_feat_name: metadata feature name to use in annotating object with whether it is positive
        or negative, default: "pair_obj_label"
        :param pair_orientation_feat_name: metadata feature name to use in annotating object with pair orientation,
        default: "pair_orientation"

        """
        assert obj_type in ["user", "utterance", "conversation"]
        self.obj_type = obj_type
        self.pairing_func = pairing_func
        self.pos_label_func = pos_label_func
        self.neg_label_func = neg_label_func
        self.pair_mode = pair_mode
        self.selector = selector
        self.pair_id_feat_name = pair_id_feat_name
        self.label_feat_name = label_feat_name
        self.pair_orientation_feat_name = pair_orientation_feat_name

    def _get_pos_neg_objects(self, corpus: Corpus):
        """
        Get positively-labelled and negatively-labelled lists of objects
        :param corpus: target Corpus
        :return: list of positive objects, list of negative objects
        """
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
        Generate a dictionary mapping the Corpus object characteristic value (i.e. pairing_func's output) to
        one positively and negatively labelled object.
        :param pos_objects: list of positively labelled objects
        :param neg_objects: list of negatively labelled objects
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

        if self.pair_mode == "first":
            return {pair_id: (pair_feat_to_pos_objs[pair_id][0],
                              pair_feat_to_neg_objs[pair_id][0])
                    for pair_id in valid_pairs}
        elif self.pair_mode == "random":
            return {pair_id: (choice(pair_feat_to_pos_objs[pair_id]),
                              choice(pair_feat_to_neg_objs[pair_id]))
                    for pair_id in valid_pairs}
        elif self.pair_mode == "maximize":
            retval = dict()
            for pair_id in valid_pairs:
                pos_objs = pair_feat_to_pos_objs[pair_id]
                neg_objs = pair_feat_to_neg_objs[pair_id]
                max_pairs = min(len(pos_objs), len(neg_objs))
                shuffle(pos_objs)
                shuffle(neg_objs)
                for idx in range(max_pairs):
                    retval[pair_id + "_" + str(idx)] = (pos_objs[idx], neg_objs[idx])
            return retval
        else:
            raise ValueError("Invalid pair_mode setting: use 'random', 'first', or 'maximize'.")

    @staticmethod
    def _assign_pair_orientations(obj_pairs):
        """
        Assigns the pair orientation (i.e. whether this pair will have a positive or negative label)
        :param obj_pairs: dictionary indexed by the paired feature instance value
        :return: dictionary of paired feature instance values to pair orientation value ('pos' or 'neg')
        """
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
        :param corpus: target Corpus
        :return: annotated Corpus
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