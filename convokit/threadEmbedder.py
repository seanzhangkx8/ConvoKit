import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .transformer import Transformer


class ThreadEmbedder(Transformer):
    """
    Transformer for embedding the thread statistics of a corpus in a
    low-dimensional space for visualization or other such purposes.

    HyperConvo.fit_transform() must be run on the Corpus first


    :param n_components: Number of dimensions to embed threads into
    :param method: Embedding method; "svd", "tsne" or "none"
    :param norm_method: Normalization method; "standard" or "none"
    :param return_components: if True, returns the components from embedding
    """

    def __init__(self, n_components=7, method="svd",
                 norm_method="standard", return_components=False):
        self.n_components = n_components
        self.method = method
        self.norm_method = norm_method
        self.return_components = return_components

    def transform(self, corpus,):
        """
        Same as fit_transform()
        """
        return self.fit_transform(corpus)

    def fit_transform(self, corpus):
        """
        :param corpus: the Corpus to use
        :return: a corpus with new meta key: "threadEmbedder",
             value: Dict, containing "X": an array with rows corresponding
             to embedded threads, "roots": an array whose ith entry is the
             thread root id of the ith row of X. If return_components is True,
             then the Dict contains a third key "components": the SVD components array
        """

        corpus_meta = corpus.get_meta()
        if "hyperconvo" not in corpus_meta:
            raise RuntimeError("Missing thread statistics: HyperConvo.fit_transform() must be run on the Corpus first")

        thread_stats = corpus_meta["hyperconvo"]
        X = []
        roots = []
        for root, feats in thread_stats.items():
            roots.append(root)
            row = np.array([v[1] if not (np.isnan(v[1]) or np.isinf(v[1])) else
                            0 for v in sorted(feats.items())])
            X.append(row)
        X = np.array(X)

        if self.norm_method.lower() == "standard":
            X = StandardScaler().fit_transform(X)
        elif self.norm_method.lower() == "none":
            pass
        else:
            raise Exception("Invalid embed_feats normalization method")

        if self.method.lower() == "svd":
            f = TruncatedSVD
        elif self.method.lower() == "tsne":
            f = TSNE
        else:
            raise Exception("Invalid embed_feats embedding method")

        emb = f(n_components=self.n_components)
        X_mid = emb.fit_transform(X) / emb.singular_values_

        retval = {"X": X_mid, "roots": roots}
        if self.return_components: retval["components"] = emb.components_

        return corpus.add_meta("threadEmbedder", retval)
