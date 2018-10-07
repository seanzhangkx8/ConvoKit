import convokit
from hyperconvo import *

corpus = convokit.Corpus("../datasets/reddit-corpus/reddit-convos.json")
hn = HyperConvo(corpus)

#print(hn.response_G.adj_out)
#print(hn.response_G.indegrees(True, True))
#print(hn.response_G.external_reciprocity_motifs())
#print(hn.response_G.dyadic_interaction_motifs())
#print(hn.response_G.incoming_triad_motifs())
print(hn.response_G.outgoing_triad_motifs())
