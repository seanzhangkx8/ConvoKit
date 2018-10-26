from collections import defaultdict, OrderedDict, Counter
import numpy as np
import scipy.stats
import itertools
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

class HyperConvo:
    def __init__(self, corpus):
        self.corpus = corpus

    def _make_hypergraph(self, uts=None, exclude_id=None):
        if uts is None:
            uts = self.corpus.utterances

        G = Hypergraph()
        username_to_utt_ids = OrderedDict()
        reply_edges = []
        speaker_to_reply_tos = defaultdict(list)
        speaker_target_pairs = set()        
        # nodes
        for _, ut in sorted(uts.items(), key=lambda u: u[1].timestamp):
            if ut.id != exclude_id:
                if ut.user not in username_to_utt_ids:
                    username_to_utt_ids[ut.user] = set()
                username_to_utt_ids[ut.user].add(ut.id)
                if ut.reply_to is not None and ut.reply_to in uts \
                    and ut.reply_to != exclude_id:
                    reply_edges.append((ut.id, ut.reply_to))
                    speaker_to_reply_tos[ut.user].append(ut.reply_to)
                    speaker_target_pairs.add((ut.user, uts[ut.reply_to].user))
                G.add_node(ut.id, info=ut.__dict__)
        # hypernodes
        for u, ids in username_to_utt_ids.items():
            G.add_hypernode(u, ids, info=u.info)
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

    def _node_type_name(self, b):
        return "C" if b else "c"

    def _degree_feats(self, uts=None, G=None, name_ext="", exclude_id=None):
        assert uts is None or G is None
        if G is None:
            G = self._make_hypergraph(uts, exclude_id=exclude_id)

        stat_funcs = {
            "max": np.max,
            "argmax": np.argmax,
            "norm.max": lambda l: np.max(l) / np.sum(l),
            "2nd-largest": lambda l: np.partition(l, -2)[-2] if len(l) > 1
                else np.nan,
            "2nd-argmax": lambda l: (-l).argsort()[1] if len(l) > 1 else np.nan,
            "norm.2nd-largest": lambda l: np.partition(l, -2)[-2] / np.sum(l)
                if len(l) > 1 else np.nan,
            "mean": np.mean,
            "mean-nonzero": lambda l: np.mean(l[l != 0]),
            "prop-nonzero": lambda l: np.mean(l != 0),
            "prop-multiple": lambda l: np.mean(l[l != 0] > 1),
            "entropy": scipy.stats.entropy,
            "2nd-largest / max": lambda l: np.partition(l, -2)[-2] / np.max(l)
                if len(l) > 1 else np.nan
        }

        stats = {}
        for from_hyper in [False, True]:
            for to_hyper in [False, True]:
                if not from_hyper and to_hyper: continue  # skip c -> C
                outdegrees = np.array(G.outdegrees(from_hyper, to_hyper))
                indegrees = np.array(G.indegrees(from_hyper, to_hyper))
                #assert sum(outdegrees)
                #assert sum(indegrees)
                for stat, stat_func in stat_funcs.items():
                    stats["{}[outdegree over {}->{} {}responses]".format(stat,
                        self._node_type_name(from_hyper),
                        self._node_type_name(to_hyper),
                        name_ext)] = stat_func(outdegrees)
                    stats["{}[indegree over {}->{} {}responses]".format(stat,
                        self._node_type_name(from_hyper),
                        self._node_type_name(to_hyper),
                        name_ext)] = stat_func(indegrees)
        return stats

    def _motif_feats(self, uts=None, G=None, name_ext="", exclude_id=None):
        assert uts is None or G is None
        if G is None:
            G = self._make_hypergraph(uts, exclude_id=exclude_id)

        stat_funcs = {
            "is-present": lambda l: len(l) > 0,
            "count": len
        }

        stats = {}
        for motif, motif_func in [
            ("reciprocity motif", G.reciprocity_motifs),
            ("external reciprocity motif", G.external_reciprocity_motifs),
            ("dyadic interaction motif", G.dyadic_interaction_motifs),
            ("incoming triads", G.incoming_triad_motifs),
            ("outgoing triads", G.outgoing_triad_motifs)]:
            motifs = motif_func()
            for stat, stat_func in stat_funcs.items():
                stats["{}[{}{}]".format(stat, motif, name_ext)] = \
                    stat_func(motifs)
        return stats

    def retrieve_feats(self, prefix_len=10, min_thread_len=10):
        threads_stats = {}
        print(len(self.corpus.utterance_threads(prefix_len=prefix_len).items()))
        for i, (root, thread) in enumerate(
            self.corpus.utterance_threads(prefix_len=prefix_len).items()):
            #if i % 100 != 0: continue
            if i % 1000 == 0: print(i)
            #if i == 10000: break
            #print(len(thread))
            if len(thread) < min_thread_len: continue
            stats = {}
            G = self._make_hypergraph(uts=thread)
            G_mid = self._make_hypergraph(uts=thread, exclude_id=root)
#            if not G_mid.edges():
#                print("BAD")
#                continue
            for k, v in self._degree_feats(G=G).items(): stats[k] = v
            for k, v in self._motif_feats(G=G).items(): stats[k] = v
            for k, v in self._degree_feats(G=G_mid,
                name_ext="mid-thread ").items(): stats[k] = v
            for k, v in self._motif_feats(G=G_mid,
                name_ext=" over mid-thread").items(): stats[k] = v
            threads_stats[root] = stats#.append(stats)
        return threads_stats

    def embed_threads(self, threads_feats, n_components=7, method="svd",
        norm_method="standard"):
        X = []#, labels = [], []
        roots = []
        for root, feats in threads_feats.items():
            roots.append(root)
            #labels.append(corpus.utterances[root].user.info["subreddit"])
            row = np.array([v[1] if not (np.isnan(v[1]) or np.isinf(v[1])) else
                0 for v in sorted(feats.items())])
            #row /= np.linalg.norm(row)
            X.append(row)
        X = np.array(X)

        if norm_method.lower() == "standard":
            X = StandardScaler().fit_transform(X)
        elif norm_method.lower() == "none":
            pass
        else:
            raise Exception("Invalid embed_feats normalization method")

        if method.lower() == "svd":
            f = TruncatedSVD
        elif method.lower() == "tsne":
            f = TSNE
        else:
            raise Exception("Invalid embed_feats embedding method")
        X_mid = f(n_components=n_components).fit_transform(X)
        return X_mid, roots

    def embed_communities(self, threads_stats, 
        community_key, n_intermediate_components=50,
        n_components=2, intermediate_method="svd", method="tsne",
        norm_method="standard",
        min_threads=10):
        X_mid, roots = self.embed_threads(threads_stats,
            n_components=n_intermediate_components, method=intermediate_method,
            norm_method=norm_method)
        X_embedded = TSNE(n_components=n_components).fit_transform(X_mid)

        labels = [self.corpus.utterances[root].user.info[community_key]
            for root in roots]
        label_counts = Counter(labels)
        subs = defaultdict(list)
        for x, label in zip(X_embedded, labels):
            if label_counts[label] >= min_threads:
                subs[label].append(x)

        labels, subs = zip(*subs.items())
        pts = [np.mean(sub / np.linalg.norm(sub), axis=0) for sub in subs]

        return pts, labels

class Hypergraph:
    def __init__(self):
        # tentatively public
        self.nodes = OrderedDict()
        self.hypernodes = OrderedDict()

        # private
        self.adj_out = OrderedDict()  # out edges for each (hyper)node
        self.adj_in = OrderedDict()   # in edges for each (hyper)node

    def add_node(self, u, info={}):
        self.nodes[u] = info
        self.adj_out[u] = OrderedDict()
        self.adj_in[u] = OrderedDict()

    def add_hypernode(self, name, nodes, info={}):
        self.hypernodes[name] = set(nodes)
        self.adj_out[name] = OrderedDict()
        self.adj_in[name] = OrderedDict()

    # edge or hyperedge
    def add_edge(self, u, v, info={}):
        assert u in self.nodes or u in self.hypernodes
        assert v in self.nodes or v in self.hypernodes
        if v not in self.adj_out[u]:
            self.adj_out[u][v] = []
        if u not in self.adj_in[v]:
            self.adj_in[v][u] = []
        self.adj_out[u][v].append(info)
        self.adj_in[v][u].append(info)

    def edges(self):
        return OrderedDict(((u, v), lst) for u, d in self.adj_out.items() 
            for v, lst in d.items())

    def outgoing_nodes(self, u):
        assert u in self.adj_out
        return OrderedDict((v, lst) for v, lst in self.adj_out[u].items()
            if v in self.nodes)

    def outgoing_hypernodes(self, u):
        assert u in self.adj_out
        return OrderedDict((v, lst) for v, lst in self.adj_out[u].items()
            if v in self.hypernodes)

    def incoming_nodes(self, v):
        assert v in self.adj_in
        return OrderedDict((u, lst) for u, lst in self.adj_in[v].items() if u in
            self.nodes)

    def incoming_hypernodes(self, v):
        assert v in self.adj_in
        return OrderedDict((u, lst) for u, lst in self.adj_in[v].items() if u in
            self.hypernodes)

    def outdegrees(self, from_hyper=False, to_hyper=False):
        return [sum([len(l) for v, l in self.adj_out[u].items() if v in
            (self.hypernodes if to_hyper else self.nodes)]) for u in
            (self.hypernodes if from_hyper else self.nodes)]

    def indegrees(self, from_hyper=False, to_hyper=False):
        return [sum([len(l) for u, l in self.adj_in[v].items() if u in
            (self.hypernodes if from_hyper else self.nodes)]) for v in
            (self.hypernodes if to_hyper else self.nodes)]

    # returns list of tuples of form (C1, c1, c2, C1->c2, c2->c1) as in paper
    def reciprocity_motifs(self):
        motifs = []
        for C1, c1_nodes in self.hypernodes.items():
            for c1 in c1_nodes:
                motifs += [(C1, c1, c2, e1, e2) for c2 in self.adj_in[c1] if
                    c2 in self.nodes and c2 in self.adj_out[C1]
                    for e1 in self.adj_out[C1][c2]
                    for e2 in self.adj_out[c2][c1]]
        return motifs

    # returns list of tuples of form (C3, c2, c1, C3->c2, c2->c1) as in paper
    def external_reciprocity_motifs(self):
        motifs = []
        for C3 in self.hypernodes:
            for c2 in self.adj_out[C3]:
                if c2 in self.nodes:
                    motifs += [(C3, c2, c1, e1, e2) for c1 in
                        set(self.adj_out[c2].keys()) - self.hypernodes[C3]
                        if c1 in self.nodes
                        for e1 in self.adj_out[C3][c2]
                        for e2 in self.adj_out[c2][c1]]
        return motifs

    # returns list of tuples of form (C1, C2, C1->C2, C2->C1) as in paper
    def dyadic_interaction_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            motifs += [(C1, C2, e1, e2) for C2 in self.adj_out[C1] if C2 in
                self.hypernodes and C1 in self.adj_out[C2]
                for e1 in self.adj_out[C1][C2]
                for e2 in self.adj_out[C2][C1]]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C2->C1, C3->C1) as in paper
    def incoming_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = list(self.adj_in[C1].keys())
            motifs += [(C1, C2, C3, e1, e2) for C2, C3 in
                itertools.combinations(incoming, 2)
                for e1 in self.adj_out[C2][C1]
                for e2 in self.adj_out[C3][C1]]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C1->C3) as in paper
    def outgoing_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            outgoing = list(self.adj_out[C1].keys())
            motifs += [(C1, C2, C3, e1, e2) for C2, C3 in
                itertools.combinations(outgoing, 2)
                for e1 in self.adj_out[C1][C2]
                for e2 in self.adj_out[C1][C3]]
        return motifs
