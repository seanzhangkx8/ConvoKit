"""Politeness Strategies features
See Section 4 of http://www.cs.cornell.edu/~cristian/Conversations_gone_awry.html
"""

import numpy as np
import pandas as pd

from collections import defaultdict

from .politeness_api.features.politeness_strategies import get_politeness_strategy_features
from .politeness_api.features.vectorizer import PolitenessFeatureVectorizer

class PolitenessStrategies:
    """
    Encapsulates extraction of politeness strategies from utterances in a
    Corpus.

    :param corpus: the corpus to compute features for.
    :type corpus: Corpus
    :param verbose: whether or not to print status messages while computing features

    :ivar corpus: the PolitenessStrategies object's corpus
    """

    def __init__(self, corpus, verbose=False):
        self.corpus = corpus
        self.verbose = verbose

        # get the comment text and corresponding IDs from the corpus
        if self.verbose: print("Retrieving comment text...")
        comment_ids = list(self.corpus.utterances.keys())
        comments = [self.corpus.utterances[cid].text for cid in comment_ids]
        # the original politeness API used in the paper was written in Python 2,
        # which uses ASCII strings. Because of this, we had to truncate Unicode
        # characters in comment text to get it to work. Although the bundled
        # API has been upgraded to Python 3, to ensure consistent results with
        # the paper we still keep the step of truncating Unicode.
        comments = [''.join([x for x in str(s) if ord(x) < 128]) for s in comments]

        # preprocess the utterances in the format expected by the API
        if self.verbose: print("Preprocessing comments...")
        comments = [{"text": comment} for comment in comments]
        processed_comments = PolitenessFeatureVectorizer.preprocess(comments)

        # use the bundled politeness API to extract politeness features for each
        # preprocessed comment
        if self.verbose: print("Extracting politeness strategies...")
        feature_dicts = [get_politeness_strategy_features(doc) for doc in processed_comments]

        # pack the extracted features into a pandas dataframe
        feature_df_raw = defaultdict(list)
        keys = [set(fd.keys()) for fd in feature_dicts]
        all_feats = set()
        for keyset in keys:
            all_feats |= keyset
        for feature_dict in feature_dicts:
            for feat in all_feats:
                feature_df_raw[feat].append(feature_dict.get(feat, np.nan))
        self.feature_df = pd.DataFrame(feature_df_raw, index=comment_ids)

    def __getitem__(self, key):
        """
        Overloaded element access operator allowing the PolitenessStrategies
        object to be used as a dictionary, with access pattern
        politenessObject[comment_id]

        :return: a dict of feature values keyed by feature name
        """
        return self.feature_df.loc[key].to_dict()
