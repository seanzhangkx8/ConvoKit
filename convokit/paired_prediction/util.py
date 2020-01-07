from convokit import Corpus
from collections import defaultdict
from random import choice, shuffle
from pandas import DataFrame
import numpy as np
from scipy.sparse import csr_matrix
from convokit.classifier.util import extract_feats_from_obj

def get_pos_neg_objects(obj_type, selector, pos_label_func, neg_label_func, corpus: Corpus):
    """
    Get positively-labelled and negatively-labelled lists of objects
    :param corpus: target Corpus
    :return: list of positive objects, list of negative objects
    """
    pos_objects = []
    neg_objects = []
    for obj in corpus.iter_objs(obj_type, selector):
        if not selector(obj): continue
        if pos_label_func(obj):
            pos_objects.append(obj)
        elif neg_label_func(obj):
            neg_objects.append(obj)
    return pos_objects, neg_objects

def pair_objs(pairing_func, pos_objects, neg_objects):
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
        pair_feat_to_pos_objs[pairing_func(obj)].append(obj)

    for obj in neg_objects:
        pair_feat_to_neg_objs[pairing_func(obj)].append(obj)

    valid_pairs = set(pair_feat_to_neg_objs).intersection(set(pair_feat_to_pos_objs))

    return {pair_id: (choice(pair_feat_to_pos_objs[pair_id]),
                      choice(pair_feat_to_neg_objs[pair_id]))
            for pair_id in valid_pairs}

def generate_bow_paired_X_y(pair_orientation_feat_name, pair_id_to_objs):
    """
    Generate the X, y matrix for paired prediction
    :param pair_id_to_objs: dictionary indexed by the paired feature instance value, with the value
    being a tuple (pos_obj, neg_obj)
    :return: X, y matrix representing the predictive features and labels respectively
    """
    pos_obj_dict = dict()
    neg_obj_dict = dict()
    for pair_id, (pos_obj, neg_obj) in pair_id_to_objs.items():
        pos_obj_dict[pair_id] = extract_feats_from_obj(pos_obj, pred_feats)
        neg_obj_dict[pair_id] = extract_feats_from_obj(neg_obj, pred_feats)
    pos_obj_df = DataFrame.from_dict(pos_obj_dict, orient='index')
    neg_obj_df = DataFrame.from_dict(neg_obj_dict, orient='index')

    X, y = [], []
    for pair_id in pair_id_to_objs:
        pos_feats = np.array(pos_obj_df.loc[pair_id])
        neg_feats = np.array(neg_obj_df.loc[pair_id])
        orientation = pair_id_to_objs[pair_id][0].meta[pair_orientation_feat_name]

        assert orientation in ["pos", "neg"]

        if orientation == "pos":
            y.append(1)
            diff = pos_feats - neg_feats
        else:
            y.append(0)
            diff = neg_feats - pos_feats

        X.append(diff)

    return csr_matrix(np.array(X)), np.array(y)


def generate_paired_X_y(pred_feats, pair_orientation_feat_name, pair_id_to_objs):
    """
    Generate the X, y matrix for paired prediction
    :param pair_id_to_objs: dictionary indexed by the paired feature instance value, with the value
    being a tuple (pos_obj, neg_obj)
    :return: X, y matrix representing the predictive features and labels respectively
    """
    pos_obj_dict = dict()
    neg_obj_dict = dict()
    for pair_id, (pos_obj, neg_obj) in pair_id_to_objs.items():
        pos_obj_dict[pair_id] = extract_feats_from_obj(pos_obj, pred_feats)
        neg_obj_dict[pair_id] = extract_feats_from_obj(neg_obj, pred_feats)
    pos_obj_df = DataFrame.from_dict(pos_obj_dict, orient='index')
    neg_obj_df = DataFrame.from_dict(neg_obj_dict, orient='index')

    X, y = [], []
    for pair_id in pair_id_to_objs:
        pos_feats = np.array(pos_obj_df.loc[pair_id])
        neg_feats = np.array(neg_obj_df.loc[pair_id])
        orientation = pair_id_to_objs[pair_id][0].meta[pair_orientation_feat_name]

        assert orientation in ["pos", "neg"]

        if orientation == "pos":
            y.append(1)
            diff = pos_feats - neg_feats
        else:
            y.append(0)
            diff = neg_feats - pos_feats

        X.append(diff)

    return csr_matrix(np.array(X)), np.array(y)

def assign_pair_orientations(obj_pairs):
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

def generate_pair_id_to_objs(corpus, obj_type, selector, pair_orientation_feat_name, label_feat_name, pair_id_feat_name):
    pair_id_to_obj = {'pos': dict(), 'neg': dict()}
    for obj in corpus.iter_objs(obj_type, selector):
        if obj.meta[pair_orientation_feat_name] is None: continue
        pair_id_to_obj[obj.meta[label_feat_name]][obj.meta[pair_id_feat_name]] = obj

    pair_ids = set(pair_id_to_obj['pos'].keys()).intersection(set(pair_id_to_obj['neg'].keys()))

    # print(set(pair_id_to_obj['pos'].keys()))
    print("Found {} valid pairs.".format(len(pair_ids)))
    pair_id_to_objs = dict()
    for pair_id in pair_ids:
        pair_id_to_objs[pair_id] = (pair_id_to_obj['pos'][pair_id], pair_id_to_obj['neg'][pair_id])
    return pair_id_to_obj