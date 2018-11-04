import convokit
import numpy as np
import matplotlib.pyplot as plt

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus"))

print("Computing hypergraph features")
hc = convokit.HyperConvo(corpus)
threads_feats = hc.retrieve_feats()

print("Computing low-dimensional embeddings")
pts, labels = hc.embed_communities(threads_feats, "subreddit",
    n_intermediate_components=50, method="tsne")

xs, ys = zip(*pts)
plt.scatter(xs, ys)
for i, txt in enumerate(labels):
    plt.annotate(txt, (xs[i], ys[i]))
plt.savefig("tsne")
plt.show()
