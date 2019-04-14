import convokit
from sklearn.neighbors import NearestNeighbors
from convokit.communityEmbedder import CommunityEmbedder

print("Loading corpus")
corpus = convokit.Corpus(filename=convokit.download("reddit-corpus"))

print("Computing hypergraph features")
hc = convokit.HyperConvo(prefix_len=10)
hc.fit_transform(corpus)

print("Computing low-dimensional embeddings")
ce = CommunityEmbedder(community_key="subreddit")
ce.fit_transform(corpus)

X_communities = corpus.get_meta()["communityEmbedder"]["pts"]
subreddits = corpus.get_meta()["communityEmbedder"]["labels"]

knn = NearestNeighbors(n_neighbors=10)
knn.fit(X_communities)

print("Nearest neighbors for each subreddit:")
for x, subreddit in zip(X_communities, subreddits):
    print(subreddit, "->", end=" ")
    for idx in knn.kneighbors([x], return_distance=False)[0][1:]:
        print(subreddits[idx], end=" ")
    print()
