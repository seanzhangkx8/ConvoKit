from convokit.model import Corpus, Conversation, Utterance, User
from typing import List, Union, Callable
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np


def extract_feats_from_obj(obj: Union[Utterance, Conversation, User], pred_feats: List[str]):
    """
    Assuming feature data has at most one level of nesting, i.e. meta['height'] = 1, and meta['grades'] = {'prelim1': 99,
    'prelim2': 75, 'final': 100}
    :param obj: Corpus object
    :param pred_feats: list of features to extract metadata from
    :return: dictionary of predictive feature names to values
    """
    retval = dict()
    for feat_name in pred_feats:
        feat_val = obj.meta[feat_name]
        if type(feat_val) == dict:
            retval.update(feat_val)
        else:
            retval[feat_name] = feat_val
    return retval


def extract_feats_dict(corpus: Corpus, obj_type: str, pred_feats: List[str],
                       selector: Callable[[Union[User, Utterance, Conversation]], bool] = lambda x: True):
    obj_id_to_feats = {obj.id: extract_feats_from_obj(obj, pred_feats) for obj in corpus.iter_objs(obj_type, selector)}

    return obj_id_to_feats


def extract_feats(corpus: Corpus, obj_type: str, pred_feats: List[str],
                  selector: Callable[[Union[User, Utterance, Conversation]], bool] = lambda x: True):
    obj_id_to_feats = extract_feats_dict(corpus, obj_type, pred_feats, selector)
    feats_df = pd.DataFrame.from_dict(obj_id_to_feats, orient='index')
    return csr_matrix(feats_df.values)


def extract_label_dict(corpus: Corpus, obj_type: str, labeller: Callable[[Union[User, Utterance, Conversation]], bool],
                       selector: Callable[[Union[User, Utterance, Conversation]], bool] = lambda x: True):

    obj_id_to_label = dict()
    for obj in corpus.iter_objs(obj_type, selector):
        obj_id_to_label[obj.id] = {'y': 1} if labeller(obj) else {'y': 0}

    return obj_id_to_label


def extract_feats_and_label(corpus: Corpus, obj_type: str, pred_feats: List[str],
                            labeller: Callable[[Union[User, Utterance, Conversation]], bool],
                            selector: Callable[[Union[User, Utterance, Conversation]], bool] = None):
    obj_id_to_feats = extract_feats_dict(corpus, obj_type, pred_feats, selector)
    obj_id_to_label = extract_label_dict(corpus, obj_type, labeller, selector)

    X_df = pd.DataFrame.from_dict(obj_id_to_feats, orient='index')
    y_df = pd.DataFrame.from_dict(obj_id_to_label, orient='index')

    X_y_df = pd.concat([X_df, y_df], axis=1, sort=False)

    y = X_y_df['y']
    X = X_y_df.drop(columns='y')

    return csr_matrix(X.values), np.array(y)






