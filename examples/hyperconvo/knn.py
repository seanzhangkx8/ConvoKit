import convokit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus"))

print("Computing hypergraph features")
hc = convokit.HyperConvo(corpus)
threads_feats = hc.fit_transform(corpus, prefix_len=10)

print("Computing low-dimensional embeddings")
X_communities, subreddits = hc.embed_communities(threads_feats, "subreddit")

knn = NearestNeighbors(n_neighbors=10)
knn.fit(X_communities)

print("Nearest neighbors for each subreddit:")
for x, subreddit in zip(X_communities, subreddits):
    print(subreddit, "->", end=" ")
    for idx in knn.kneighbors([x], return_distance=False)[0][1:]:
        print(subreddits[idx], end=" ")
    print()
