import convokit
import numpy as np
import matplotlib.pyplot as plt

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus"))

print("Computing hypergraph features")
hc = convokit.HyperConvo()
hc.fit_transform(corpus, prefix_len=10)
threads_feats = corpus.get_meta()["hyperconvo"]
feat_names = list(sorted(threads_feats[list(threads_feats.keys())[0]].keys()))

print("Computing low-dimensional embeddings")
te = convokit.ThreadEmbedder()
te.fit_transform(corpus, return_components=True)
X_threads = corpus.get_meta()["threadEmbedder"]["X"]
roots = corpus.get_meta()["threadEmbedder"]["roots"]
components = corpus.get_meta()["threadEmbedder"]["components"]

ce = convokit.CommunityEmbedder()
ce.fit_transform(corpus, community_key="subreddit")
X_communities = corpus.get_meta()["communityEmbedder"]["pts"]
subreddits = corpus.get_meta()["communityEmbedder"]["labels"]

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
