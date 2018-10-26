import convokit
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# create corpus object
corpus = convokit.Corpus("../../datasets/reddit-corpus/reddit-convos.json")
hc = convokit.HyperConvo(corpus)

threads_feats = hc.retrieve_feats(prefix_len=10)

X, labels = [], []
for root, feats in threads_feats.items():
    labels.append(corpus.utterances[root].user.info["subreddit"])
    row = np.array([v[1] if not (np.isnan(v[1]) or np.isinf(v[1])) else 0
        for v in sorted(feats.items())])
    feat_names = list(sorted(feats.keys()))
    #row /= np.linalg.norm(row)
    X.append(row)
X = np.array(X)
X = StandardScaler().fit_transform(X)

c = Counter(labels)

svd = TruncatedSVD(n_components=7)
X_mid = svd.fit_transform(X).tolist()
subs = defaultdict(list)
for x, label in zip(X_mid, labels):
    if c[label] >= 50:
        subs[label].append(x)

labels, subs = zip(*subs.items())
X_f = np.array([np.mean(sub / np.linalg.norm(sub), axis=0) for sub in subs])

print("TOP SUBREDDITS")
for d in range(7):
    print("dimension {}".format(d))
    print("- worst subreddits")
    ranked = list(sorted(zip(labels, X_f), key=lambda x: x[1][d]))
    for label, x in ranked[:10]:
        print("\t{}  {:.4f}".format(label, x[d]))
    print("- best subreddits")
    for label, x in reversed(ranked[-10:]):
        print("\t{}  {:.4f}".format(label, x[d]))
    print()
    print()

print()
print("TOP FEATURES")
for d in range(7):
    print("dimension {}".format(d))
    print("- most negative features")
    ranked = list(sorted(zip(feat_names, np.transpose(svd.components_)),
        key=lambda x: x[1][d]))
    for label, x in ranked[:10]:
        print("\t{}  {:.4f}".format(label, x[d]))
    print("- most positive features")
    for label, x in reversed(ranked[-10:]):
        print("\t{}  {:.4f}".format(label, x[d]))
    print()
    print()
