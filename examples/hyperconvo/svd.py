import convokit
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

corpus = convokit.Corpus(filename=convokit.download("reddit-corpus"))
hc = convokit.HyperConvo(corpus)

threads_feats = hc.retrieve_feats(prefix_len=10)
feat_names = list(sorted(threads_feats[list(threads_feats.keys())[0]].keys()))

X_threads, roots, components = hc.embed_threads(threads_feats,
    return_components=True)
X_communities, subreddits = hc.embed_communities(threads_feats, "subreddit")

print("TOP THREADS")
for d in range(7):
    print("dimension {}".format(d))
    print("- worst threads")
    ranked = list(sorted(zip(roots, X_threads), key=lambda x: x[1][d]))
    for label, x in ranked[:10]:
        print("\t{}  {:.4f}".format(label, x[d]))
    print("- best threads")
    for label, x in reversed(ranked[-10:]):
        print("\t{}  {:.4f}".format(label, x[d]))
    print()
    print()

print("TOP SUBREDDITS")
for d in range(7):
    print("dimension {}".format(d))
    print("- worst subreddits")
    ranked = list(sorted(zip(subreddits, X_communities), key=lambda x: x[1][d]))
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
    ranked = list(sorted(zip(feat_names, np.transpose(components)),
        key=lambda x: x[1][d]))
    for label, x in ranked[:10]:
        print("\t{}  {:.4f}".format(label, x[d]))
    print("- most positive features")
    for label, x in reversed(ranked[-10:]):
        print("\t{}  {:.4f}".format(label, x[d]))
    print()
    print()
