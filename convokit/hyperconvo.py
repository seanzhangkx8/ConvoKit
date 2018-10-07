from collections import defaultdict

class HyperConvo:
    def __init__(self, corpus):
        self.corpus = corpus

        self.response_G = self._make_response_hypergraph()

    def _make_response_hypergraph(self):
        G = Hypergraph()
        self.username_to_utt_ids = defaultdict(list)
        reply_edges = []
        speaker_to_reply_tos = defaultdict(list)
        speaker_target_pairs = set()        
        # nodes
        for ut in self.corpus.utterances.values():
            self.username_to_utt_ids[ut.user].append(ut.id)
            if ut.reply_to is not None:
                reply_edges.append((ut.id, ut.reply_to))
                speaker_to_reply_tos[ut.user].append(ut.reply_to)
                speaker_target_pairs.add((ut.user,
                    self.corpus.utterances[ut.reply_to].user))
            G.add_node(ut.id, info={
                "text": ut.text,
                "timestamp": ut.timestamp,
                "reply-to": ut.reply_to,
                "root": ut.root,
                "user": ut.user
            })
        # hypernodes
        for u in self.corpus.users():
            G.add_hypernode(u, self.username_to_utt_ids[u],
                info=u.info)
        # reply edges
        for u, v in reply_edges:
            G.add_edge(u, v)
        # user to utterance response edges
        for u, reply_tos in speaker_to_reply_tos.items():
            for reply_to in reply_tos:
                G.add_edge(u, reply_to)
        # user to user response edges
        for u, v in speaker_target_pairs:
            G.add_edge(u, v)
        return G

class Hypergraph:
    def __init__(self):
        # tentatively public
        self.nodes = {}
        self.hypernodes = {}

        # private
        self.adj_out = {}  # out edges for each (hyper)node
        self.adj_in = {}   # in edges for each (hyper)node

    def add_node(self, u, info={}):
        self.nodes[u] = info
        self.adj_out[u] = {}
        self.adj_in[u] = {}

    def add_hypernode(self, name, nodes, info={}):
        self.hypernodes[name] = set(nodes)
        self.adj_out[name] = {}
        self.adj_in[name] = {}

    # edge or hyperedge
    def add_edge(self, u, v, info={}):
        assert u in self.nodes or u in self.hypernodes
        assert v in self.nodes or v in self.hypernodes
        self.adj_out[u][v] = info
        self.adj_in[v][u] = info

    def edges(self):
        return {(u, v): info for u, d in self.adj_out.items() for v, info in
            d.items()}

    def outgoing_nodes(self, u):
        assert u in self.adj_out
        return {v: info for v, info in self.adj_out[u].items() if v in
            self.nodes}

    def outgoing_hypernodes(self, u):
        assert u in self.adj_out
        return {v: info for v, info in self.adj_out[u].items() if v in
            self.hypernodes}

    def incoming_nodes(self, v):
        assert v in self.adj_in
        return {u: info for u, info in self.adj_in[v].items() if u in
            self.nodes}

    def incoming_hypernodes(self, v):
        assert v in self.adj_in
        return {u: info for u, info in self.adj_in[v].items() if u in
            self.hypernodes}

    def outdegrees(self, from_hyper=False, to_hyper=False):
        return [len([v for v in self.adj_out[u] if v in
            (self.hypernodes if to_hyper else self.nodes)]) for u in
            (self.hypernodes if from_hyper else self.nodes)]

    def indegrees(self, from_hyper=False, to_hyper=False):
        return [len([u for u in self.adj_in[v] if u in
            (self.hypernodes if from_hyper else self.nodes)]) for v in
            (self.hypernodes if to_hyper else self.nodes)]

    # returns list of tuples of form (C1, c1, c2) as in paper
    def reciprocity_motifs(self):
        motifs = []
        for C1, c1_nodes in self.hypernodes.items():
            for c1 in c1_nodes:
                motifs += [(C1, c1, c2) for c2 in self.adj_in[c1] if
                    c2 in self.nodes and c2 in self.adj_out[C1]]
        return motifs

    # returns list of tuples of form (C3, c2, c1) as in paper
    def external_reciprocity_motifs(self):
        motifs = []
        for C3 in self.hypernodes:
            for c2 in self.adj_out[C3]:
                if c2 in self.nodes:
                    motifs += [(C3, c2, c1) for c1 in
                        set(self.adj_out[c2].keys()) - self.hypernodes[C3]
                        if c1 in self.nodes]
        return motifs

    # returns list of tuples of form (C1, C2) as in paper
    def dyadic_interaction_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            motifs += [(C1, C2) for C2 in self.adj_out[C1] if C2 in
                self.hypernodes and C1 in self.adj_out[C2]]
        return motifs

    # returns list of tuples of form (C1, C2, C3) as in paper
    def incoming_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = list(self.adj_in[C1].keys())
            motifs += [(C1, C2, C3) for C2, C3 in zip(incoming[::2],
                incoming[1::2])]
        return motifs

    # returns list of tuples of form (C1, C2, C3) as in paper
    def outgoing_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            outgoing = list(self.adj_out[C1].keys())
            motifs += [(C1, C2, C3) for C2, C3 in zip(outgoing[::2],
                outgoing[1::2])]
        return motifs
