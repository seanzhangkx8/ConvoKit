from convokit.model import Corpus, Conversation, Utterance
from typing import List, Hashable

def extract_convo_features(convo: Conversation, pred_feats: List[Hashable]):
    """
    DESIGN DECISION: how to extract values from nested dicts? consider flattening nested dict # TODO
    :param convo:
    :param pred_feats:
    :return:
    """
    retval = dict()
    for feat_name in pred_feats:
        feat_val = convo.meta[feat_name]
        if type(feat_val) == dict:
            retval.update(feat_val)
        else:
            retval[feat_name] = feat_val

    return retval