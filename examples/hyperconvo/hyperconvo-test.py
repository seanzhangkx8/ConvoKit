import convokit
import numpy as np

corpus = convokit.Corpus("../../datasets/reddit-corpus/reddit-convos.json")
hc = convokit.HyperConvo(corpus)

def summarize_dist(name, l):
    print("{}: min {}, mean {:.4f}, max {}".format(
        name, min(l), np.mean(l), max(l)))

summarize_dist("user to user indegrees", hc.G.indegrees(True, True))
summarize_dist("user to user outdegrees", hc.G.outdegrees(True, True))
summarize_dist("user to comment indegrees", hc.G.indegrees(True, False))
summarize_dist("user to comment outdegrees", hc.G.outdegrees(True, False))
summarize_dist("comment to comment indegrees", hc.G.indegrees(False, False))
summarize_dist("comment to comment outdegrees", hc.G.outdegrees(False, False))
print()

def summarize_motifs(name, l):
    print("{}: count {}".format(name, len(l)))

summarize_motifs("reciprocity", hc.G.reciprocity_motifs())
summarize_motifs("external reciprocity", hc.G.external_reciprocity_motifs())
summarize_motifs("dyadic interaction", hc.G.dyadic_interaction_motifs())
summarize_motifs("incoming triad", hc.G.incoming_triad_motifs())
summarize_motifs("outgoing triad", hc.G.outgoing_triad_motifs())
print()

print("example reciprocity motif:", hc.G.reciprocity_motifs()[0])
