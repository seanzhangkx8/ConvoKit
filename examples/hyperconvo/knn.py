import convokit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

corpus = convokit.Corpus(filename=convokit.download("reddit-corpus"))
hc = convokit.HyperConvo(corpus)

threads_feats = hc.retrieve_feats(prefix_len=10)
feat_names = list(sorted(threads_feats[list(threads_feats.keys())[0]].keys()))

X_communities, subreddits = hc.embed_communities(threads_feats, "subreddit")

knn = NearestNeighbors(n_neighbors=10)
knn.fit(X_communities)

for x, subreddit in zip(X_communities, subreddits):
    print(subreddit, "->", end=" ")
    for idx in knn.kneighbors([x], return_distance=False)[0][1:]:
        print(subreddits[idx], end=" ")
    print()
