import convokit
import numpy as np
import matplotlib.pyplot as plt

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus-small"))

print("Computing hypergraph features")
hc = convokit.HyperConvo()
hc.fit_transform(corpus)

print("Computing low-dimensional embeddings")
pts, labels = hc.embed_communities("subreddit",
              n_intermediate_components=7, method="tsne")

xs, ys = zip(*pts)
plt.scatter(xs, ys)
for i, txt in enumerate(labels):
    plt.annotate(txt, (xs[i], ys[i]))
plt.savefig("tsne")
plt.show()
