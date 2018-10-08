import convokit
import numpy as np

# create corpus object
corpus = convokit.Corpus("../../datasets/reddit-corpus/reddit-convos.json")
hc = convokit.HyperConvo(corpus)

# we might not need to expose make_hypergraph publicly, but this gives us a 
#   hypergraph to demonstrate Hypergraph methods with
G = hc.make_hypergraph()

def summarize_dist(name, l):
    print("{}: min {}, mean {:.4f}, max {}".format(
        name, min(l), np.mean(l), max(l)))

# in- and outdegree distributions
summarize_dist("user to user indegrees", G.indegrees(True, True))
summarize_dist("user to user outdegrees", G.outdegrees(True, True))
summarize_dist("user to comment indegrees", G.indegrees(True, False))
summarize_dist("user to comment outdegrees", G.outdegrees(True, False))
summarize_dist("comment to comment indegrees", G.indegrees(False, False))
summarize_dist("comment to comment outdegrees", G.outdegrees(False, False))
print()

def summarize_motifs(name, l):
    print("{}: count {}".format(name, len(l)))

# motif extraction
summarize_motifs("reciprocity", G.reciprocity_motifs())
summarize_motifs("external reciprocity", G.external_reciprocity_motifs())
summarize_motifs("dyadic interaction", G.dyadic_interaction_motifs())
summarize_motifs("incoming triad", G.incoming_triad_motifs())
summarize_motifs("outgoing triad", G.outgoing_triad_motifs())
print()

print("example reciprocity motif:", G.reciprocity_motifs()[0])
print()

# HyperConvo interface: get high-level degree features
feats = hc.degree_feats(G=G)   # G=G is an optional speedup: avoid recomputing G
for k, v in feats.items():
    print("{}: {:.4f}".format(k, v))
