from collections import defaultdict

class HyperConvo:
    def __init__(self, corpus):
        self.corpus = corpus
        self.G = Hypergraph()

        self.username_to_utt_ids = defaultdict(list)
        reply_edges = []
        speaker_to_reply_tos = defaultdict(list)
        speaker_target_pairs = set()        
        # nodes
        for ut in self.corpus.utterances.values():
            self.username_to_utt_ids[ut.user.name].append(ut.id)
            if ut.reply_to is not None:
                self.reply_edges.append((ut.id, ut.reply_to))
                speaker_to_reply_tos[ut.user.name].append(ut.reply_to)
                speaker_target_pairs.add((ut.user.name,
                    self.corpus.utterances[ut.reply_to].user.name))
            self.G.add_node(ut.id, info={
                "text": ut.text,
                "timestamp": ut.timestamp,
                "reply-to": ut.reply_to,
                "root": ut.root,
                "user": ut.user.name
            })
        # hypernodes
        for u in self.corpus.users():
            self.G.add_hypernode(u.name, self.username_to_utt_ids[u.name],
                info=u.info)
        # reply edges
        for u, v in reply_edges:
            self.G.add_edge(u, v, info={"type": "reply"})
        # user to utterance response edges
        for u, reply_tos in speaker_to_reply_tos.items():
            for reply_to in reply_tos:
                self.G.add_edge(u, reply_to)
        # user to user response edges
        for u, v in speaker_target_pairs:
            self.G.add_edge(u, v)

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

    def add_hypernode(self, name, nodes, info={}):
        self.hypernodes[name] = set(nodes)

    # edge or hyperedge
    def add_edge(self, u, v, info={}):
        assert u in self.nodes or u in self.hypernodes
        assert v in self.nodes or v in self.hypernodes
        if u not in self.adj_out:
            self.adj_out[u] = {}
        self.adj_out[u][v] = info

        if v not in self.adj_in:
            self.adj_in[v] = {}
        self.adj_in[v][u] = info

    def edges(self):
        return {(u, v): info for u, d in self.adj_out.items() for v, info in
            d.items()}

    def out_edges_to_nodes(self, u):
        assert u in self.adj_out
        return {v: info for v, info in self.adj_out[u].items() if v in
            self.nodes}

    def out_edges_to_hypernodes(self, u):
        assert u in self.adj_out
        return {v: info for v, info in self.adj_out[u].items() if v in
            self.hypernodes}

    def in_edges_from_nodes(self, v):
        assert v in self.adj_in
        return {u: info for u, info in self.adj_in[v].items() if u in
            self.nodes}

    def in_edges_from_hypernodes(self, v):
        assert v in self.adj_in
        return {u: info for u, info in self.adj_in[v].items() if u in
            self.hypernodes}
