"""Implements the hypergraph conversation model from
http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html."""

import itertools
from collections import defaultdict, OrderedDict
from enum import Enum, auto
from typing import Tuple, Dict

import numpy as np
import scipy.stats
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .transformer import Transformer


class HyperConvo(Transformer):
    """Encapsulates computation of hypergraph features for a particular
    corpus.

    General workflow: first, retrieve features from the corpus conversational
    threads using retrieve_feats. Then, either use the features directly, or
    use the convenience methods embed_threads and embed_communities to embed
    threads or communities respectively in a low-dimensional space for further
    analysis or visualization.

    As features, we compute the degree distribution statistics from Table 4 of
    http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html,
    for both a whole conversation and its midthread, and for indegree and
    outdegree distributions of C->C, C->c and c->c edges, as in the paper.
    We also compute the presence and count of each motif type specified in Fig 2.
    However, we do not include features making use of reaction edges, due to our
    inability to release the Facebook data used in the paper (which reaction
    edges are most naturally suited for). In particular, we do not include edge
    distribution statistics from Table 4, as these rely on the presence of
    reaction edges. We hope to implement a more general version of these
    reaction features in an upcoming release.

    :param corpus: the corpus to compute features for.
    :type corpus: Corpus

    :ivar corpus: the coordination object's corpus.
    """

    def __init__(self, corpus):
        self.corpus = corpus
        self.threads_feats = None

    def transform(self, corpus, prefix_len=10, min_thread_len=10):
        return self.fit_transform(corpus, prefix_len=prefix_len, min_thread_len=min_thread_len)

    def fit_transform(self, corpus, prefix_len=10, min_thread_len=10):
        return self.retrieve_feats(prefix_len=prefix_len, min_thread_len = min_thread_len)

    def _make_hypergraph(self, uts=None, exclude_id=None):
        if uts is None:
            uts = self.corpus.utterances

        G = Hypergraph()
        username_to_utt_ids = OrderedDict()
        reply_edges = []
        speaker_to_reply_tos = defaultdict(list)
        speaker_target_pairs = set()
        # nodes
        for _, ut in sorted(uts.items(), key=lambda h: h[1].timestamp):
            if ut.id != exclude_id:
                if ut.user not in username_to_utt_ids:
                    username_to_utt_ids[ut.user] = set()
                username_to_utt_ids[ut.user].add(ut.id)
                if ut.reply_to is not None and ut.reply_to in uts \
                        and ut.reply_to != exclude_id:
                    reply_edges.append((ut.id, ut.reply_to))
                    speaker_to_reply_tos[ut.user].append(ut.reply_to)
                    speaker_target_pairs.add((ut.user, uts[ut.reply_to].user, ut.timestamp))
                G.add_node(ut.id, info=ut.__dict__)
        # hypernodes
        for u, ids in username_to_utt_ids.items():
            G.add_hypernode(u, ids, info=u.meta)
        # reply edges
        for u, v in reply_edges:
            # print("ADDING TIMESTAMP")
            G.add_edge(u, v)
        # user to utterance response edges
        for u, reply_tos in speaker_to_reply_tos.items():
            for reply_to in reply_tos:
                G.add_edge(u, reply_to)
        # user to user response edges
        for u, v, timestamp in speaker_target_pairs:
            G.add_edge(u, v, {'timestamp': timestamp})
        return G

    @staticmethod
    def _node_type_name(b):
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
                                                                         HyperConvo._node_type_name(from_hyper),
                                                                         HyperConvo._node_type_name(to_hyper),
                                                                         name_ext)] = stat_func(outdegrees)
                    stats["{}[indegree over {}->{} {}responses]".format(stat,
                                                                        HyperConvo._node_type_name(from_hyper),
                                                                        HyperConvo._node_type_name(to_hyper),
                                                                        name_ext)] = stat_func(indegrees)
        return stats

    def _get_motifs(self, uts=None, G=None, exclude_id=None):
        assert uts is None or G is None
        if G is None:
            G = self._make_hypergraph(uts, exclude_id=exclude_id)

        motifs = dict()

        for motif_type, motif_func in [
            # ("reciprocity motif", G.reciprocity_motifs),
            # ("external reciprocity motif", G.external_reciprocity_motifs),
            # ("dyadic interaction motif", G.dyadic_interaction_motifs),
            (MotifType.NO_EDGE_TRIADS.name, G.no_edge_triad_motifs),
            (MotifType.SINGLE_EDGE_TRIADS.name, G.single_edge_triad_motifs),
            (MotifType.INCOMING_TRIADS.name, G.incoming_triad_motifs),
            (MotifType.OUTGOING_TRIADS.name, G.outgoing_triad_motifs),
            (MotifType.DYADIC_TRIADS.name, G.dyadic_triad_motifs),
            (MotifType.UNIDIRECTIONAL_TRIADS.name, G.unidirectional_triad_motifs),
            (MotifType.INCOMING_2TO3_TRIADS.name, G.incoming_2to3_triad_motifs),
            (MotifType.INCOMING_1TO3_TRIADS.name, G.incoming_1to3_triad_motifs),
            (MotifType.DIRECTED_CYCLE_TRIADS.name, G.directed_cycle_triad_motifs),
            (MotifType.OUTGOING_3TO1_TRIADS.name, G.outgoing_3to1_triad_motifs),
            (MotifType.INCOMING_RECIPROCAL_TRIADS.name, G.incoming_reciprocal_triad_motifs),
            (MotifType.OUTGOING_RECIPROCAL_TRIADS.name, G.outgoing_reciprocal_motifs),
            (MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name, G.directed_cycle_1to3_triad_motifs),
            (MotifType.DIRECIPROCAL_TRIADS.name, G.direciprocal_triad_motifs),
            (MotifType.DIRECIPROCAL_2TO3_TRIADS.name, G.direciprocal_2to3_triad_motifs),
            (MotifType.TRIRECIPROCAL_TRIADS, G.trireciprocal_triad_motifs)
        ]:
            motifs[motif_type] = motif_func()

        return motifs

    @staticmethod
    def probabilities(transitions: Dict):
        """
        Takes a transitions count dictionary Dict[(MotifType.name->MotifType.name)->Int]
        :return: transitions probability dictionary Dict[(MotifType.name->MotifType.name)->Float]
        """
        probs = dict()

        for parent, children in TriadMotif.relations().items():
            total = sum(transitions[(parent, c)] for c in children) + transitions[(parent, parent)]

            probs[(parent, parent)] = (transitions[(parent, parent)] / total) if total > 0 else 0
            for c in children:
                probs[(parent, c)] = (transitions[(parent, c)] / total) if total > 0 else 0

        return probs

    @staticmethod
    def _latent_motif_count(motifs, prob):
        """
        Takes a dictionary of (MotifType.name, List[Motif]) and a bool prob, indicating whether
        transition probabilities need to be returned

        :return: Returns a tuple of a dictionary of latent motif counts
        and a dictionary of motif->motif transition probabilities
         (Dict[MotifType.name->Int], Dict[(MotifType.name->MotifType.name)->Float])
         The second element is None if prob=False
        """
        latent_motif_count = {motif_type.name: 0 for motif_type in MotifType}

        transitions = TriadMotif.transitions()
        for motif_type, motif_instances in motifs.items():
            for motif_instance in motif_instances:
                curr_motif = motif_instance
                child_motif_type = curr_motif.get_type()
                # Reflexive edge
                transitions[(child_motif_type, child_motif_type)] += 1

                # print(transitions)
                while True:
                    latent_motif_count[curr_motif.get_type()] +=  1
                    curr_motif = curr_motif.regress()
                    if curr_motif is None: break
                    parent_motif_type = curr_motif.get_type()
                    transitions[(parent_motif_type, child_motif_type)] += 1
                    child_motif_type = parent_motif_type

        probs = HyperConvo.probabilities(transitions) if prob else None

        return latent_motif_count, probs

    def _motif_feats(self, latent=True, prob=True, uts=None, G=None, name_ext="", exclude_id=None):
        if prob is True: # prob can only be True if latent is True
            assert latent is True

        motifs = self._get_motifs(uts, G, exclude_id)

        stat_funcs = {
            "is-present": lambda l: len(l) > 0,
            "count": len
        }

        stats = {}

        for motif_type in motifs:
            for stat, stat_func in stat_funcs.items():
                stats["{}[{}{}]".format(stat, str(motif_type), name_ext)] = \
                    stat_func(motifs[motif_type])

        if latent:
            latent_motif_count, probabilities = HyperConvo._latent_motif_count(motifs, prob=prob)
            for motif_type in latent_motif_count:
                stats["is-present[LATENT_{}{}]".format(motif_type, name_ext)] = \
                    (latent_motif_count[motif_type] > 0)
                stats["count[LATENT_{}{}]".format(motif_type, name_ext)] = latent_motif_count[motif_type]

            if prob:
                assert probabilities is not None
                for p, v in probabilities.items():
                    stats["prob[{}]".format(p)] = v

        return stats

    def retrieve_feats(self, prefix_len=10, min_thread_len=10):
        threads_stats = {}
        """Retrieve all hypergraph features for a given corpus (viewed as a set
        of conversation threads).

        :param prefix_len: Length (in number of utterances) of each thread to
            consider when constructing its hypergraph
        :param min_thread_len: Only consider threads of at least this length

        :return: A dictionary from a thread root id to its stats dictionary,
            which is a dictionary from feature names to feature values.
        """
        for i, (root, thread) in enumerate(
                self.corpus.utterance_threads(prefix_len=prefix_len).items()):
            #if i % 100 != 0: continue
            #if i % 1000 == 0: print(i)
            #if i == 10000: break
            if len(thread) < min_thread_len: continue
            stats = {}
            G = self._make_hypergraph(uts=thread)
            G_mid = self._make_hypergraph(uts=thread, exclude_id=root)
            #            print("EDGES")
            #            print(G.hypernodes)
            #            print("MID-THREAD EDGES")
            #            print(G_mid.hypernodes)
            #            print("DIFF")
            #            print(set(G.hypernodes.keys()) - set(G_mid.hypernodes.keys()))
            #            print(set(G.nodes.keys()) - set(G_mid.nodes.keys()))
            #            input()
            #            a, b = self._degree_feats(G=G), self._degree_feats(G=G_mid)
            #            print([a[k] - b[k] for k in a])
            #            input()
            for k, v in self._degree_feats(G=G).items(): stats[k] = v
            for k, v in self._motif_feats(prob=True, G=G).items(): stats[k] = v
            for k, v in self._degree_feats(G=G_mid,
                                           name_ext="mid-thread ").items(): stats[k] = v
            for k, v in self._motif_feats(prob=False, G=G_mid,
                                          name_ext=" over mid-thread").items(): stats[k] = v
            threads_stats[root] = stats#.append(stats)
        return threads_stats

    @staticmethod
    def embed_threads(threads_feats, n_components=7, method="svd",
                      norm_method="standard", return_components=False):
        """Convenience method to embed the output of retrieve_feats in a
        low-dimensional space.

        :param threads_feats: Output of retrieve_feats
        :param n_components: Number of dimensions to embed into
        :param method: embedding method; either "svd" or "tsne"
        :param norm_method: data normalization method; either "standard"
            (normalize each feature to 0 mean and 1 variance) or "none"
        :param return_components: if using SVD method, whether to output
            SVD components array

        :return: a tuple (X, roots) where X is an array with rows corresponding
            to embedded threads, and roots is an array whose ith entry is the
            thread root id of the ith row of X. If return_components is True,
            then the tuple contains a third entry, the SVD components array
        """

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
        emb = f(n_components=n_components)
        X_mid = emb.fit_transform(X) / emb.singular_values_
        #print("SINGULAR VALUES")
        #print(emb.singular_values_)

        if not return_components:
            return X_mid, roots
        else:
            return X_mid, roots, emb.components_

    def embed_communities(self, threads_stats,
                          community_key, n_intermediate_components=7,
                          n_components=2, intermediate_method="svd", method="none",
                          norm_method="standard"):
        """Convenience method to embed the output of retrieve_feats in a
        low-dimensional space, and group threads together into communities
        in this space.

        :param threads_stats: Output of retrieve_feats
        :param community_key: Key in "user-info" dictionary of each utterance
            whose corresponding value we'll use as the community label for that
            utterance
        :param n_intermediate_components: Number of dimensions to embed threads into
        :param intermediate_method: Embedding method for threads
            (see embed_threads)
        :param n_components: Number of dimensions to embed communities into
        :param method: Embedding method; "svd", "tsne" or "none"
        :param norm_method: Data normalization method; either "standard"
            (normalize each feature to 0 mean and 1 variance) or "none"

        :return: a tuple (X, labels) where X is an array with rows corresponding
            to embedded communities, and labels is an array whose ith entry is
            the community of the ith row of X.
        """

        X_mid, roots = HyperConvo.embed_threads(threads_stats,
                                                n_components=n_intermediate_components, method=intermediate_method,
                                                norm_method=norm_method)

        if method.lower() == "svd":
            f = TruncatedSVD
        elif method.lower() == "tsne":
            f = TSNE
        elif method.lower() == "none":
            f = None
        else:
            raise Exception("Invalid embed_communities embedding method")
        if f is not None:
            X_embedded = f(n_components=n_components).fit_transform(X_mid)
        else:
            X_embedded = X_mid

        labels = [self.corpus.utterances[root].user.info[community_key]
                  for root in roots]
        # label_counts = Counter(labels)
        subs = defaultdict(list)
        for x, label in zip(X_embedded, labels):
            subs[label].append(x / np.linalg.norm(x))

        labels, subs = zip(*subs.items())
        pts = [np.mean(sub, axis=0) for sub in subs]

        return pts, labels



class Hypergraph:
    """Represents a hypergraph, consisting of nodes, directed edges,
    hypernodes (each of which is a set of nodes) and hyperedges (directed edges
    from hypernodes to hypernodes). Contains functionality to extract motifs
    from hypergraphs (Fig 2 of
    http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html)
    """
    def __init__(self):
        # public
        self.nodes = OrderedDict()
        self.hypernodes = OrderedDict()

        # private
        self.adj_out = OrderedDict()  # out edges for each (hyper)node
        self.adj_in = OrderedDict()   # in edges for each (hyper)node

    def add_node(self, u, info=dict()):
        self.nodes[u] = info
        self.adj_out[u] = OrderedDict()
        self.adj_in[u] = OrderedDict()

    def add_hypernode(self, name, nodes, info=dict()):
        self.hypernodes[name] = set(nodes)
        self.adj_out[name] = OrderedDict()
        self.adj_in[name] = OrderedDict()

    # edge or hyperedge
    def add_edge(self, u, v, info=dict()):
        assert u in self.nodes or u in self.hypernodes
        assert v in self.nodes or v in self.hypernodes
        if u in self.hypernodes and v in self.hypernodes:
            assert len(info.keys()) > 0
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

    # returns list of tuples of form (C1, C2, C3), no edges
    def no_edge_triad_motifs(self):
        motifs = []
        for C1, C2, C3 in itertools.combinations(self.hypernodes, 3):
            if C1 not in self.adj_in[C2] and C1 not in self.adj_in[C3]:
                if C2 not in self.adj_in[C3] and C2 not in self.adj_in[C1]:
                    if C3 not in self.adj_in[C1] and C3 not in self.adj_in[C2]:
                        motifs += [TriadMotif((C1, C2, C3), (), MotifType.NO_EDGE_TRIADS.name)]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2)
    def single_edge_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            outgoing = set(self.outgoing_hypernodes(C1))
            incoming = set(self.incoming_hypernodes(C1))
            non_adjacent = set(self.hypernodes) - outgoing.union(incoming)
            outgoing_only = outgoing - incoming

            # for C2 in outgoing_only:
            #     assert C2 not in self.adj_in[C1]
            #     for C3 in non_adjacent:
            #         if C3 in self.adj_out[C2]: continue
            #         if C3 in self.adj_in[C2]: continue
            #         assert C3 not in self.adj_out[C1]
            #         assert C3 not in self.adj_in[C1]
            #         motifs += [(C1, C2, C3, e1) for e1 in self.adj_out[C1][C2]]

            motifs += [TriadMotif((C1, C2, C3), (e1,), MotifType.SINGLE_EDGE_TRIADS.name) for C2 in outgoing_only
                       for C3 in non_adjacent if ((C3 not in self.adj_out[C2]) and (C3 not in self.adj_in[C2]))
                       for e1 in self.adj_out[C1][C2]]
        # for m in motifs: print(m)
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C1)
    def dyadic_triad_motifs(self):
        motifs = []
        for C3 in self.hypernodes: # define the triad with respect to C3 <- prevents double counting
            outgoing = set(self.outgoing_hypernodes(C3))
            incoming = set(self.incoming_hypernodes(C3))
            non_adjacent = set(self.hypernodes) - outgoing.union(incoming)

            motifs += [TriadMotif((C1, C2, C3), (e1, e2), MotifType.DYADIC_TRIADS.name) for C1, C2 in itertools.combinations(non_adjacent, 2)
                       for e1 in self.adj_out[C1].get(C2, [])
                       for e2 in self.adj_out[C2].get(C1, [])]
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
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            incoming_only = incoming - outgoing
            motifs += [TriadMotif((C1, C2, C3), (e1, e2), MotifType.INCOMING_TRIADS.name) for C2, C3 in
                       itertools.combinations(incoming_only, 2)
                       for e1 in self.adj_out[C2][C1]
                       for e2 in self.adj_out[C3][C1]]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C1->C3) as in paper
    def outgoing_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            outgoing = set(self.outgoing_hypernodes(C1))
            incoming = set(self.incoming_hypernodes(C1))
            outgoing_only = outgoing - incoming
            motifs += [TriadMotif((C1, C2, C3), (e1, e2), MotifType.OUTGOING_TRIADS.name) for C2, C3 in
                       itertools.combinations(outgoing_only, 2)
                       for e1 in self.adj_out[C1][C2]
                       for e2 in self.adj_out[C1][C3]]
        # for m in motifs: #BUGGY?
        #     if m[0] is m[1]:
        #         print(m)
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C3)
    def unidirectional_triad_motifs(self):
        motifs = []
        for C2 in self.hypernodes: # define the motif with respect to C2
            incoming = set(self.adj_in[C2].keys())
            outgoing = set(n for n in self.adj_out[C2].keys() if n in self.hypernodes)
            incoming_only = incoming - outgoing # ignore edges C2->C1
            outgoing_only = outgoing - incoming # ignore edges C3->C2
            for C1 in incoming_only:
                for C3 in outgoing_only:
                    # ensure C3 and C1 have no edges between them
                    if len(self.adj_out[C3].get(C1, [])) > 0 or len(self.adj_out[C1].get(C3, [])) > 0: continue
                    motifs += [TriadMotif((C1, C2, C3), (e1, e2), MotifType.UNIDIRECTIONAL_TRIADS.name)
                               for e1 in self.adj_out[C1][C2]
                               for e2 in self.adj_out[C2][C3]]

        return motifs

    # returns list of tuples of form (C1, C2, C3, C2->C1, C3->C1, C2->C3)
    def incoming_2to3_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = self.incoming_hypernodes(C1)
            for C2, C3 in itertools.combinations(incoming, 2):
                if len(self.adj_out[C3].get(C2, [])) > 0: continue # ensure no C3->C2
                motifs += [TriadMotif((C1, C2, C3), (e1, e2, e3), MotifType.INCOMING_2TO3_TRIADS.name)
                           for e3 in self.adj_out[C2].get(C3, []) # safe get because e3 might not exist
                           for e1 in self.adj_out[C2][C1]
                           for e2 in self.adj_out[C3][C1]
                           ]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C3, C3->C1)
    def directed_cycle_triad_motifs(self):
        # not efficient
        motifs = []
        for C1, C2, C3 in itertools.combinations(self.hypernodes, 3):
            if len(self.adj_out[C1].get(C3, [])) > 0: continue
            if len(self.adj_out[C2].get(C1, [])) > 0: continue
            if len(self.adj_out[C3].get(C2, [])) > 0: continue
            motifs += [TriadMotif((C1, C2, C3), (e1, e2, e3), MotifType.DIRECTED_CYCLE_TRIADS.name)
                       for e1 in self.adj_out[C1].get(C2, [])
                       for e2 in self.adj_out[C2].get(C3, [])
                       for e3 in self.adj_out[C3].get(C1, [])]

        return motifs

    # returns list of tuples of form (C1, C2, C3, C2->C1, C3->C1, C1->C3)
    def incoming_1to3_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = self.incoming_hypernodes(C1)
            for C2, C3 in itertools.combinations(incoming, 2):
                if len(self.adj_out[C1].get(C2, [])) > 0: continue
                if len(self.adj_out[C3].get(C2, [])) > 0: continue
                if len(self.adj_out[C2].get(C3, [])) > 0: continue

                motifs += [TriadMotif((C1, C2, C3), (e1, e2, e3), MotifType.INCOMING_1TO3_TRIADS.name)
                           for e3 in self.adj_out[C1].get(C3, [])
                           for e1 in self.adj_out[C2][C1]
                           for e2 in self.adj_out[C3][C1]
                           ]

        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C1->C3, C3->C1)
    def outgoing_3to1_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            outgoing = self.outgoing_hypernodes(C1)
            for C2, C3 in itertools.combinations(outgoing, 2):
                if len(self.adj_out[C2].get(C1, [])) > 0: continue
                if len(self.adj_out[C3].get(C2, [])) > 0: continue
                if len(self.adj_out[C2].get(C3, [])) > 0: continue

                motifs += [TriadMotif((C1, C2, C3), (e1, e2, e3), MotifType.OUTGOING_3TO1_TRIADS.name)
                           for e3 in self.adj_out[C1].get(C3, [])
                           for e1 in self.adj_out[C1][C2]
                           for e2 in self.adj_out[C1][C3]
                           ]

        return motifs

    # returns list of tuples of form (C1, C2, C3, C2->C1, C3->C1, C2->C3, C3->C2)
    def incoming_reciprocal_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            incoming_only = incoming - outgoing
            motifs += [TriadMotif((C1, C2, C3), (e1, e2, e3, e4), MotifType.INCOMING_RECIPROCAL_TRIADS.name) for C2, C3 in
                       itertools.combinations(incoming_only, 2)
                       for e3 in self.adj_out[C2].get(C3, [])
                       for e4 in self.adj_out[C3].get(C2, [])
                       for e1 in self.adj_out[C2][C1]
                       for e2 in self.adj_out[C3][C1]
                       ]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C1->C3, C2->C3, C3->C2)
    def outgoing_reciprocal_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            outgoing_only = outgoing - incoming
            motifs += [TriadMotif((C1, C2, C3), (e1, e2, e3, e4), MotifType.OUTGOING_RECIPROCAL_TRIADS.name) for C2, C3 in
                       itertools.combinations(outgoing_only, 2)
                       for e3 in self.adj_out[C2].get(C3, [])
                       for e4 in self.adj_out[C3].get(C2, [])
                       for e1 in self.adj_out[C1][C2]
                       for e2 in self.adj_out[C1][C3]
                       ]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C3, C3->C1, C1->C3)
    def directed_cycle_1to3_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            outgoing = set(self.outgoing_hypernodes(C1))
            for C2, C3 in itertools.combinations(outgoing, 2):
                if len(self.adj_out[C2].get(C1, [])) > 0: continue
                if len(self.adj_out[C3].get(C2, [])) > 0: continue
                motifs += [TriadMotif((C1, C2, C3), (e1, e2, e3, e4), MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name)
                           for e2 in self.adj_out[C2].get(C3, [])
                           for e3 in self.adj_out[C3].get(C1, [])
                           for e1 in self.adj_out[C1][C2]
                           for e4 in self.adj_out[C1][C3]
                           ]
        # for m in motifs:
        #     print(m)
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1)
    def direciprocal_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            in_and_out = incoming.intersection(outgoing)
            for C2, C3 in itertools.combinations(in_and_out, 2):
                if len(self.adj_out[C2].get(C3, [])) > 0: continue
                if len(self.adj_out[C3].get(C2, [])) > 0: continue

                motifs += [TriadMotif((C1, C2, C3), (e1, e2, e3, e4), MotifType.DIRECIPROCAL_TRIADS.name)
                           for e1 in self.adj_out[C1][C2]
                           for e2 in self.adj_out[C2][C1]
                           for e3 in self.adj_out[C1][C3]
                           for e4 in self.adj_out[C3][C1]
                           ]
        return motifs

    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1, C2->C3)
    def direciprocal_2to3_triad_motifs(self):
        motifs = []
        for C1 in self.hypernodes:
            incoming = set(self.incoming_hypernodes(C1))
            outgoing = set(self.outgoing_hypernodes(C1))
            in_and_out = incoming.intersection(outgoing)
            for C2, C3 in itertools.combinations(in_and_out, 2):
                if len(self.adj_out[C2].get(C3, [])) > 0: continue
                if len(self.adj_out[C3].get(C2, [])) > 0: continue

                motifs += [TriadMotif((C1, C2, C3), (e1, e2, e3, e4, e5), MotifType.DIRECIPROCAL_2TO3_TRIADS.name)
                           for e5 in self.adj_out[C2].get(C3, [])
                           for e1 in self.adj_out[C1][C2]
                           for e2 in self.adj_out[C2][C1]
                           for e3 in self.adj_out[C1][C3]
                           for e4 in self.adj_out[C3][C1]
                           ]
        return motifs


    # returns list of tuples of form (C1, C2, C3, C1->C2, C2->C1, C2->C3, C3->C2, C3->C1, C1->C3)
    def trireciprocal_triad_motifs(self):
        # prevents triple-counting
        motifs = [TriadMotif((C1, C2, C3), (e1, e2, e3, e4, e5, e6), MotifType.TRIRECIPROCAL_TRIADS.name)
                  for C1, C2, C3 in itertools.combinations(self.hypernodes, 3)
                  for e1 in self.adj_out[C1].get(C2, [])
                  for e2 in self.adj_out[C2].get(C1, [])
                  for e3 in self.adj_out[C2].get(C3, [])
                  for e4 in self.adj_out[C3].get(C2, [])
                  for e5 in self.adj_out[C3].get(C1, [])
                  for e6 in self.adj_out[C1].get(C3, [])
                  ]

        return motifs


class MotifType(Enum):
    NO_EDGE_TRIADS = auto()
    SINGLE_EDGE_TRIADS = auto()
    INCOMING_TRIADS = auto()
    OUTGOING_TRIADS = auto()
    DYADIC_TRIADS = auto()
    UNIDIRECTIONAL_TRIADS = auto()
    INCOMING_2TO3_TRIADS = auto()
    INCOMING_1TO3_TRIADS = auto()
    DIRECTED_CYCLE_TRIADS = auto()
    OUTGOING_3TO1_TRIADS = auto()
    INCOMING_RECIPROCAL_TRIADS = auto()
    OUTGOING_RECIPROCAL_TRIADS = auto()
    DIRECTED_CYCLE_1TO3_TRIADS = auto()
    DIRECIPROCAL_TRIADS = auto()
    DIRECIPROCAL_2TO3_TRIADS = auto()
    TRIRECIPROCAL_TRIADS = auto()


class TriadMotif:
    """Represents a *TRIAD* motif, consisting of hypernodes and directed edges
    Contains functionality to temporally regress a motif to its antecedent stages
    """
    def __init__(self, hypernodes: Tuple, edges: Tuple, triad_type: str):
        self.hypernodes = hypernodes
        self.edges = edges
        self.triad_type = triad_type

    def get_hypernodes(self):
        return self.hypernodes

    def get_edges(self):
        return self.edges

    def get_type(self):
        return self.triad_type

    def _last_added_edge_idx(self):
        edges = self.edges
        max_idx = 0
        max_timestamp = 0
        # print("The current motif type is: {}".format(self.type))
        # print("This is what the edge set looks like: {}".format(edges))
        for i, e in enumerate(edges):
            # print(e)
            if int(e['timestamp']) > max_timestamp:
                max_idx = i
                max_timestamp = int(e['timestamp'])
        return max_idx

    # returns a motif with the last edge removed
    def regress(self):
        if self.triad_type == MotifType.NO_EDGE_TRIADS.name: return None
        last_edge_idx = self._last_added_edge_idx()
        remaining_edges = self.edges[:last_edge_idx] + self.edges[last_edge_idx+1:]
        # print("Remaining edges are: {}".format(remaining_edges))
        new_type = TriadMotif.regression()[self.triad_type][last_edge_idx]
        # print("New type is: {}".format(new_type))
        return TriadMotif(self.hypernodes, remaining_edges, new_type)


    @staticmethod
    def regression():
        """
        :return: dictionary where key is MotifType, and value is dictionary of motif types that result from deletion
        # of specified edge number
        """
        return {
            MotifType.NO_EDGE_TRIADS.name: {},

            MotifType.SINGLE_EDGE_TRIADS.name: {0: MotifType.NO_EDGE_TRIADS.name},

            MotifType.INCOMING_TRIADS.name: {0: MotifType.SINGLE_EDGE_TRIADS.name, 1: MotifType.SINGLE_EDGE_TRIADS.name},
            MotifType.OUTGOING_TRIADS.name: {0: MotifType.SINGLE_EDGE_TRIADS.name, 1: MotifType.SINGLE_EDGE_TRIADS.name},
            MotifType.DYADIC_TRIADS.name: {0: MotifType.SINGLE_EDGE_TRIADS.name, 1: MotifType.SINGLE_EDGE_TRIADS.name},
            MotifType.UNIDIRECTIONAL_TRIADS.name: {0: MotifType.SINGLE_EDGE_TRIADS.name, 1: MotifType.SINGLE_EDGE_TRIADS.name},

            # (C1, C2, C3, C2->C1, C3->C1, C2->C3)
            MotifType.INCOMING_2TO3_TRIADS.name: {0: MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                  1: MotifType.OUTGOING_TRIADS.name,
                                                  2: MotifType.INCOMING_TRIADS.name},

            # (C1, C2, C3, C1->C2, C2->C3, C3->C1)
            MotifType.DIRECTED_CYCLE_TRIADS.name: {0: MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                   1: MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                   2: MotifType.UNIDIRECTIONAL_TRIADS.name},

            # (C1, C2, C3, C2->C1, C3->C1, C1->C3)
            MotifType.INCOMING_1TO3_TRIADS.name: {0: MotifType.DYADIC_TRIADS.name,
                                                  1: MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                  2: MotifType.INCOMING_TRIADS.name},

            # (C1, C2, C3, C1->C2, C1->C3, C3->C1)
            MotifType.OUTGOING_3TO1_TRIADS.name: {0: MotifType.DYADIC_TRIADS.name,
                                                  1: MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                  2: MotifType.OUTGOING_TRIADS.name},

            # (C1, C2, C3, C2->C1, C3->C1, C2->C3, C3->C2)
            MotifType.INCOMING_RECIPROCAL_TRIADS.name: {0: MotifType.OUTGOING_3TO1_TRIADS.name,
                                                        1: MotifType.OUTGOING_3TO1_TRIADS.name,
                                                        2: MotifType.INCOMING_2TO3_TRIADS.name,
                                                        3: MotifType.INCOMING_2TO3_TRIADS.name},

            # (C1, C2, C3, C1->C2, C1->C3, C2->C3, C3->C2)
            MotifType.OUTGOING_RECIPROCAL_TRIADS.name: {0: MotifType.INCOMING_1TO3_TRIADS.name,
                                                        1: MotifType.INCOMING_1TO3_TRIADS.name,
                                                        2: MotifType.INCOMING_2TO3_TRIADS.name,
                                                        3: MotifType.INCOMING_2TO3_TRIADS.name},

            # (C1, C2, C3, C1->C2, C2->C3, C3->C1, C1->C3)
            MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name: {0: MotifType.INCOMING_1TO3_TRIADS.name,
                                                        1: MotifType.OUTGOING_3TO1_TRIADS.name,
                                                        2: MotifType.INCOMING_2TO3_TRIADS.name,
                                                        3: MotifType.DIRECTED_CYCLE_TRIADS.name},

            # (C1, C2, C3, C1->C2, C2->C1, C1->C3, C3->C1)
            MotifType.DIRECIPROCAL_TRIADS.name: {0: MotifType.INCOMING_1TO3_TRIADS.name,
                                                 1: MotifType.OUTGOING_3TO1_TRIADS.name,
                                                 2: MotifType.INCOMING_1TO3_TRIADS.name,
                                                 3: MotifType.OUTGOING_3TO1_TRIADS.name},

            # (C1, C2, C3, C1->C2, C1->C3, C2->C1, C3->C1, C2->C3)
            MotifType.DIRECIPROCAL_2TO3_TRIADS.name: {0: MotifType.OUTGOING_RECIPROCAL_TRIADS.name,
                                                      1: MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name,
                                                      2: MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name,
                                                      3: MotifType.INCOMING_RECIPROCAL_TRIADS.name,
                                                      4: MotifType.DIRECIPROCAL_TRIADS.name},

            MotifType.TRIRECIPROCAL_TRIADS.name: {0: MotifType.DIRECIPROCAL_2TO3_TRIADS.name,
                                                  1: MotifType.DIRECIPROCAL_2TO3_TRIADS.name,
                                                  2: MotifType.DIRECIPROCAL_2TO3_TRIADS.name,
                                                  3: MotifType.DIRECIPROCAL_2TO3_TRIADS.name,
                                                  4: MotifType.DIRECIPROCAL_2TO3_TRIADS.name,
                                                  5: MotifType.DIRECIPROCAL_2TO3_TRIADS.name}
        }

    @staticmethod
    def relations():
        """
        :return: dictionary where key is parent motif name, and value is a list of child motif names
        """
        return {
            MotifType.NO_EDGE_TRIADS.name: [MotifType.SINGLE_EDGE_TRIADS.name],
            MotifType.SINGLE_EDGE_TRIADS.name: [MotifType.UNIDIRECTIONAL_TRIADS.name,
                                                MotifType.DYADIC_TRIADS.name,
                                                MotifType.OUTGOING_TRIADS.name,
                                                MotifType.INCOMING_TRIADS.name],
            MotifType.UNIDIRECTIONAL_TRIADS.name: [MotifType.OUTGOING_3TO1_TRIADS.name,
                                                   MotifType.DIRECTED_CYCLE_TRIADS.name,
                                                   MotifType.INCOMING_1TO3_TRIADS.name,
                                                   MotifType.INCOMING_2TO3_TRIADS.name],
            MotifType.DYADIC_TRIADS.name: [MotifType.OUTGOING_3TO1_TRIADS.name,
                                           MotifType.INCOMING_1TO3_TRIADS.name],
            MotifType.OUTGOING_TRIADS.name: [MotifType.OUTGOING_3TO1_TRIADS.name,
                                             MotifType.INCOMING_2TO3_TRIADS.name],
            MotifType.INCOMING_TRIADS.name: [MotifType.INCOMING_1TO3_TRIADS.name,
                                             MotifType.INCOMING_2TO3_TRIADS.name],
            MotifType.OUTGOING_3TO1_TRIADS.name: [MotifType.DIRECIPROCAL_TRIADS.name,
                                                  MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name,
                                                  MotifType.INCOMING_RECIPROCAL_TRIADS.name],
            MotifType.DIRECTED_CYCLE_TRIADS.name: [MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name],
            MotifType.INCOMING_1TO3_TRIADS.name: [MotifType.DIRECIPROCAL_TRIADS.name,
                                                  MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name,
                                                  MotifType.OUTGOING_RECIPROCAL_TRIADS.name],
            MotifType.INCOMING_2TO3_TRIADS.name: [MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name,
                                                  MotifType.OUTGOING_RECIPROCAL_TRIADS.name,
                                                  MotifType.INCOMING_RECIPROCAL_TRIADS.name],
            MotifType.DIRECIPROCAL_TRIADS.name: [MotifType.DIRECIPROCAL_2TO3_TRIADS.name],
            MotifType.DIRECTED_CYCLE_1TO3_TRIADS.name: [MotifType.DIRECIPROCAL_2TO3_TRIADS.name],
            MotifType.OUTGOING_RECIPROCAL_TRIADS.name: [MotifType.DIRECIPROCAL_2TO3_TRIADS.name],
            MotifType.INCOMING_RECIPROCAL_TRIADS.name: [MotifType.DIRECIPROCAL_2TO3_TRIADS.name],
            MotifType.DIRECIPROCAL_2TO3_TRIADS.name: [MotifType.TRIRECIPROCAL_TRIADS.name],
            MotifType.TRIRECIPROCAL_TRIADS.name: []
        }

    @staticmethod
    def transitions():
        """
        :return: Dictionary of Key: Motif transition (MotifType.name, MotifType.name), Value: # of transitions (Int)
                with all values set to 0
        """
        retval = dict()

        for parent, children in TriadMotif.relations().items():
            retval[(parent, parent)] = 0
            for c in children:
                retval[(parent, c)] = 0

        return retval
