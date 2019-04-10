import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from transformer import Transformer


class ThreadEmbedder(Transformer):
    """
    Transformer for embedding the thread statistics of a corpus in a
    low-dimensional space for visualization or other such purposes.

    HyperConvo.fit_transform() must be run on the Corpus first

    :return: a corpus with new meta key: "threadEmbedder",
             value: Dict, containing "X": an array with rows corresponding
             to embedded threads, "roots": an array whose ith entry is the
             thread root id of the ith row of X. If return_components is True,
             then the Dict contains a third key "components": the SVD components array
    """

    def __init__(self):
        pass

    def transform(self, corpus, n_components=7, method="svd",
                  norm_method="standard", return_components=False):
        """
        Same as fit_transform()
        """
        return self.fit_transform(corpus, n_components=n_components, method=method,
                                  norm_method=norm_method, return_components=return_components)

    def fit_transform(self, corpus, n_components=7, method="svd",
                      norm_method="standard", return_components=False):
        """
        :param corpus: Corpus object
        :param n_components: Number of dimensions to embed into
        :param method: embedding method; either "svd" or "tsne"
        :param norm_method: data normalization method; either "standard"
            (normalize each feature to 0 mean and 1 variance) or "none"
        :param return_components: if using SVD method, whether to output
            SVD components array
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

        if norm_method.lower() == "standard":
            X = StandardScaler().fit_transform(X)
        elif norm_method.lower() == "none":
            pass
        else:
            raise Exception("Invalid embed_feats normalization method")

        if method.lower() == "svd":
            f = TruncatedSVD
        elif method.lower() == "tsne":
            f = TSNE
        else:
            raise Exception("Invalid embed_feats embedding method")

        emb = f(n_components=n_components)
        X_mid = emb.fit_transform(X) / emb.singular_values_

        retval = {"X": X_mid, "roots": roots}
        if return_components: retval["components"] = emb.components_

        return corpus.add_meta("threadEmbedder", retval)
