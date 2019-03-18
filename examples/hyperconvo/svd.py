import convokit
import numpy as np
import matplotlib.pyplot as plt

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus"))

print("Computing hypergraph features")
hc = convokit.HyperConvo()
threads_feats = hc.fit_transform(corpus, prefix_len=10)
feat_names = list(sorted(threads_feats[list(threads_feats.keys())[0]].keys()))

print("Computing low-dimensional embeddings")
X_threads, roots, components = hc.embed_threads(return_components=True)
X_communities, subreddits = hc.embed_communities("subreddit")

print("TOP THREADS")
for d in range(7):
    print("dimension {}".format(d))
    print("- most-negative threads")
    ranked = list(sorted(zip(roots, X_threads), key=lambda x: x[1][d]))
    for label, x in ranked[:10]:
        print("\t{}  {:.4f}".format(label, x[d]))
    print("- most-positive threads")
    for label, x in reversed(ranked[-10:]):
        print("\t{}  {:.4f}".format(label, x[d]))
    print()
    print()

print("TOP SUBREDDITS")
for d in range(7):
    print("dimension {}".format(d))
    print("- most-negative subreddits")
    ranked = list(sorted(zip(subreddits, X_communities), key=lambda x: x[1][d]))
    for label, x in ranked[:10]:
        print("\t{}  {:.4f}".format(label, x[d]))
    print("- most-positive subreddits")
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
