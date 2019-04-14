import convokit
import numpy as np
import matplotlib.pyplot as plt
from threadEmbedder import ThreadEmbedder
from communityEmbedder import CommunityEmbedder

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

print("Computing hypergraph features")
hc = convokit.HyperConvo()
hc.fit_transform(corpus)

print("Computing low-dimensional embeddings")
te = ThreadEmbedder(n_components=7)
te.fit_transform(corpus)

ce = CommunityEmbedder(community_key="subreddit", method="tsne")
ce.fit_transform(corpus)

pts = corpus.get_meta()["communityEmbedder"]["pts"]
labels = corpus.get_meta()["communityEmbedder"]["labels"]

xs, ys = zip(*pts)
plt.scatter(xs, ys)
for i, txt in enumerate(labels):
    plt.annotate(txt, (xs[i], ys[i]))
plt.savefig("tsne")
plt.show()
