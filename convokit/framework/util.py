from convokit.model import Corpus, Conversation, Utterance, User
from typing import List, Hashable, Union, Callable
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

def extract_feats_from_obj(obj: Union[Utterance, Conversation, User], pred_feats: List[Hashable]):
    """
    DESIGN DECISION: how to extract values from nested dicts? consider flattening nested dict? # TODO
    :param obj:
    :param pred_feats:
    :return:
    """
    retval = dict()
    for feat_name in pred_feats:
        feat_val = obj.meta[feat_name]
        if type(feat_val) == dict:
            retval.update(feat_val)
        else:
            retval[feat_name] = feat_val
    return retval

def extract_feats_dict(corpus: Corpus, obj_type: str, pred_feats: List[Hashable]):
    obj_iters = {"conversation": corpus.iter_conversations,
                 "user": corpus.iter_users,
                 "utterance": corpus.iter_utterances}

    obj_id_to_feats = {obj.id: extract_feats_from_obj(obj, pred_feats) for obj in obj_iters[obj_type]()}

    return obj_id_to_feats

def extract_feats(corpus: Corpus, obj_type: str, pred_feats: List[Hashable]):
    obj_id_to_feats = extract_feats_dict(corpus, obj_type, pred_feats)
    feats_df = pd.DataFrame.from_dict(obj_id_to_feats, orient='index')
    return csr_matrix(feats_df.values)

def extract_label_dict(corpus: Corpus, obj_type: str, y_func: Callable[[Union[User, Utterance, Conversation]], bool]):
    obj_iters = {"conversation": corpus.iter_conversations,
                 "user": corpus.iter_users,
                 "utterance": corpus.iter_utterances}

    obj_id_to_label = dict()
    for obj in obj_iters[obj_type]():
        obj_id_to_label[obj.id] = {'y': 1} if y_func(obj) else {'y': 0}

    return obj_id_to_label

def extract_feats_and_label(corpus: Corpus, obj_type: str, pred_feats: List[Hashable],
                        y_func: Callable[[Union[User, Utterance, Conversation]], bool]):
    obj_id_to_feats = extract_feats_dict(corpus, obj_type, pred_feats)
    obj_id_to_label = extract_label_dict(corpus, obj_type, y_func)

    X_df = pd.DataFrame.from_dict(obj_id_to_feats, orient='index')
    y_df = pd.DataFrame.from_dict(obj_id_to_label, orient='index')

    X_y_df = pd.concat([X_df, y_df], axis=1)

    y = X_y_df['y']
    X = X_y_df.drop(columns='y')

    return csr_matrix(X.values), np.array(y)






