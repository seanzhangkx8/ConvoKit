import convokit
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from collections import Counter

corpus = convokit.Corpus("../../datasets/reddit-corpus/reddit-convos.json")
hc = convokit.HyperConvo(corpus)

threads_feats = hc.retrieve_feats()

pts, labels = hc.embed_communities(threads_feats, "subreddit", min_threads=50,
    method="tsne")

xs, ys = zip(*pts)
plt.scatter(xs, ys)#, color=colors)
for i, txt in enumerate(labels):
    plt.annotate(txt, (xs[i], ys[i]))
plt.savefig("tsne")
plt.show()
