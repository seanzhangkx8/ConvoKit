import convokit
import numpy as np
import matplotlib.pyplot as plt

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

print("Computing hypergraph features")
hc = convokit.HyperConvo()
hc.fit_transform(corpus)

print("Computing low-dimensional embeddings")
te = convokit.ThreadEmbedder()
te.fit_transform(corpus, n_components=7)

ce = convokit.CommunityEmbedder()
ce.fit_transform(corpus, community_key="subreddit", method="tsne")

pts = corpus.get_meta()["communityEmbedder"]["pts"]
labels = corpus.get_meta()["communityEmbedder"]["labels"]

xs, ys = zip(*pts)
plt.scatter(xs, ys)
for i, txt in enumerate(labels):
    plt.annotate(txt, (xs[i], ys[i]))
plt.savefig("tsne")
plt.show()
