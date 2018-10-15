import convokit
import numpy as np
from sklearn.manifold import TSNE
from collections import defaultdict
import matplotlib.pyplot as plt

# create corpus object
corpus = convokit.Corpus("../../datasets/reddit-corpus/reddit-convos.json")
hc = convokit.HyperConvo(corpus)

subreddits = defaultdict(dict)
for ut in corpus.utterances.values():
    subreddit = corpus.utterances[ut.root].user.info["subreddit"]
    subreddits[subreddit][ut.id] = ut

X, labels = [], []
for name, subreddit in subreddits.items():
    feats = hc.all_feats(uts=subreddit)
    labels.append(name)
    X.append(np.array([v[1] for v in sorted(feats.items())]))
X = np.array(X)

X_embedded = TSNE(n_components=2).fit_transform(X)

xs, ys = zip(*X_embedded)
plt.scatter(xs, ys)
for i, txt in enumerate(labels):
    plt.annotate(txt, (xs[i], ys[i]))
plt.show()
