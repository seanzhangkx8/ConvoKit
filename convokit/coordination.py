"""Coordination features
(https://www.cs.cornell.edu/~cristian/Echoes_of_power.html)."""

import pkg_resources
import re
from collections import defaultdict

CoordinationWordCategories = ["article", "auxverb", "conj", "adverb",
        "ppron", "ipron", "preps", "quant"]

class CoordinationScore(dict):
    def scores_for_marker(self, marker):
        return {speaker: scores[marker] for speaker, scores in self.items()}

    def averages_by_user(self):
        return {speaker: sum(scores.values()) / len(scores)
                for speaker, scores in self.items()}

    def averages_by_marker(self, strict_thresh=False):
        self.precompute_aggregates()
        return self.a1_avg_by_marker if strict_thresh else self.avg_by_marker

    def aggregate(self, method=3):
        assert 1 <= method and method <= 3
        self.precompute_aggregates()
        if method == 1:
            return self.agg1
        elif method == 2:
            return self.agg2
        else:
            return self.agg3

    # helper functions
    def precompute_aggregates(self):
        a1_scores_by_marker = defaultdict(list)
        scores_by_marker = defaultdict(list)
        for speaker, scores in self.items():
            for cat, score in scores.items():
                scores_by_marker[cat].append(score)
                if len(scores) == len(CoordinationWordCategories):
                    a1_scores_by_marker[cat].append(score)
        do_agg2 = False
        if len(scores_by_marker) == len(CoordinationWordCategories):
            do_agg2 = True
            avg_score_by_marker = {cat: sum(scores) / len(scores) 
                    for cat, scores in scores_by_marker.items()}
        agg1s, agg2s, agg3s = [], [], []
        for speaker, scoredict in self.items():
            scores = list(scoredict.values())
            if len(scores) >= 1:
                avg = sum(scores) / len(scores)
                agg3s.append(avg)
                if len(scores) == len(CoordinationWordCategories):
                    agg1s.append(avg)
                if do_agg2:
                    for cat in avg_score_by_marker:
                        if cat not in scoredict:
                            scores.append(avg_score_by_marker[cat])
                    agg2s.append(sum(scores) / len(scores))
        agg1 = sum(agg1s) / len(agg1s) if agg1s else None
        agg2 = sum(agg2s) / len(agg2s) if agg2s else None  
        agg3 = sum(agg3s) / len(agg3s) if agg3s else None

        a1_avg_by_marker = {cat: sum(scores) / len(scores)
                for cat, scores in a1_scores_by_marker.items()}
        avg_by_marker = {cat: sum(scores) / len(scores)
                for cat, scores in scores_by_marker.items()}
        self.precomputed_aggregates = True
        self.a1_avg_by_marker = a1_avg_by_marker
        self.avg_by_marker = avg_by_marker
        self.agg1 = agg1
        self.agg2 = agg2
        self.agg3 = agg3

class Coordination:
    """Encapsulates computation of coordination-based features for a particular
    corpus.

    :param corpus: the corpus to compute features for.
    :type corpus: Corpus

    :ivar corpus: the coordination object's corpus. 
    """

    def __init__(self, corpus):
        self.corpus = corpus
        self.precomputed = False

    def precompute(self):
        """Call this to run the time-consuming annotation process explicitly.
        For example, this lets you save the annotated coordination object as a
        pickle to cache the precomputation results."""

        if not self.precomputed:
            self.compute_liwc_reverse_dict()
            self.annot_liwc_cats()
            self.precomputed = True

    def score(self, speakers, group, speaker_thresh=0, target_thresh=3,
            utterances_thresh=0, speaker_thresh_indiv=0, target_thresh_indiv=0,
            utterances_thresh_indiv=0, utterance_thresh_func=None):
        """Computes the coordination scores for each speaker, given a set of
        speakers and a group of targets.

        :param speakers: A collection of usernames or user objects corresponding
            to the speakers we want to compute scores for.
        :param group: A collection of usernames or user objects corresponding to
            the group of targets.
        :param speaker_thresh: Thresholds based on
            minimum number of times the speaker uses each coordination
            marker.
        :type speaker_thresh: int

        :param target_thresh: Thresholds based on
            minimum number of times the target uses each coordination
            marker.
        :type target_thresh: int
        :param utterances_thresh: Thresholds based on
            the minimum number of utterances for each speaker.
        :type utterances_thresh: int
        :param speaker_thresh_indiv: Like
            `speaker_thresh` but only considers the utterances
            between a speaker and a single target; thresholds whether the
            utterances for a single target should be considered for a
            particular speaker.
        :type speaker_thresh_indiv: int
        :param target_thresh_indiv: Like
            `target_thresh` but thresholds whether a single target's
            utterances should be considered for a particular speaker.
        :type target_thresh_indiv: int
        :param utterances_thresh_indiv: Like
            `utterances_thresh` but thresholds whether a single target's
            utterances should be considered for a particular speaker.
        :type utterances_thresh_indiv: int
        :param utterance_thresh_func: Optional utterance-level
            threshold function that takes in a speaker `Utterance` and
            the `Utterance` the speaker replied to, and returns a `bool`
            corresponding to whether or not to include the utterance in
            scoring.
        :type utterance_thresh_func: function

        :return:
            A dictionary of scores:

            ::

                {
                    speaker_1: { dictionary of scores by coordination marker },
                    speaker_2: scores,
                    ...
                }

            The keys are of the same types as the input: if a username was 
            passed in, the corresponding key will be a username, etc.
        """
        self.precompute()
        speakers = set(speakers)
        group = set(group)

        # are we referring to speakers by name or user obj?
        any_speaker = next(iter(speakers))
        fine_grained_speakers = not isinstance(any_speaker, str)

        # are we referring to targets by name or user obj?
        any_target = next(iter(group))
        fine_grained_targets = not isinstance(any_target, str)

        utterances = []
        for u in self.corpus.utterances.values():
            speaker = u.user if fine_grained_speakers else u.user.name
            if speaker in speakers:
                if u.reply_to is not None:
                    reply_to = self.corpus.utterances[u.reply_to]
                    target = reply_to.user if fine_grained_targets else \
                            reply_to.user.name
                    if target in group:
                        utterances.append(u)
        return self.scores_over_utterances(speakers, utterances,
                speaker_thresh, target_thresh, utterances_thresh,
                speaker_thresh_indiv, target_thresh_indiv,
                utterances_thresh_indiv, utterance_thresh_func,
                fine_grained_speakers, fine_grained_targets)

    def pairwise_scores(self, pairs, speaker_thresh=0, target_thresh=3,
            utterances_thresh=0, speaker_thresh_indiv=0, target_thresh_indiv=0,
            utterances_thresh_indiv=0, utterance_thresh_func=None):
        """Computes all pairwise coordination scores given a collection of
        (speaker, target) pairs.
        
        :param pairs: collection of (speaker, target) pairs where
            each speaker and target can be either a username or a user
            object.
        :type pairs: Collection
        
        Also accepted: all threshold arguments accepted by :func:`score()`.

        :return:
            Dictionary of scores indexed by (speaker, target) pairs.

            Each value is itself a dictionary with scores indexed by
            coordination marker.
        """
        self.precompute()
        pairs = set(pairs)
        any_speaker = next(iter(pairs))[0]
        if isinstance(any_speaker, str):
            pairs_utts = self.corpus.pairwise_exchanges(lambda x, y:
                    (x.name, y.name) in pairs, user_names_only=True)
        else:
            pairs_utts = self.corpus.pairwise_exchanges(lambda x, y:
                    (x, y) in pairs, user_names_only=False)
        all_scores = CoordinationScore()
        for (speaker, target), utterances in pairs_utts.items():
            scores = self.scores_over_utterances([speaker], utterances,
                    speaker_thresh, target_thresh, utterances_thresh,
                    speaker_thresh_indiv, target_thresh_indiv,
                    utterances_thresh_indiv, utterance_thresh_func,
                    not isinstance(speaker, str), not isinstance(target, str))
            for m in scores.values():
                all_scores[speaker, target] = m
        return all_scores

    def score_report(self, scores):
        """Create a "score report" of aggregate scores given a score output
        produced by `score` or `pairwise_scores`.

        - Aggregate 1: average scores only over users with a score for each
            coordination marker.
        - Aggregate 2: fill in missing scores for a user by using the group
            score for each missing marker. (Assumes different people in a group
            coordinate the same way.)
        - Aggregate 3: fill in missing scores for a user by using the average
            score over the markers we can compute coordination for for that 
            user. (Assumes a user coordinates the same way across different
            coordination markers.)

        :param scores: Scores to produce a report for.
        :type scores: dict

        :return: A tuple (marker_a1, marker, agg1, agg2, agg3):

            - marker_a1 is a dictionary of aggregate scores by marker,
                using the scores only over users included in Aggregate 1.
            - marker is a dictionary of aggregate scores by marker,
                using the scores of all users with a coordination score for
                that marker.
            - agg1, agg2 and agg3 are Aggregate 1, 2 and 3 scores respectively.

        """
        marker_a1 = scores.averages_by_marker(strict_thresh=True)  
        marker = scores.averages_by_marker()
        agg1 = scores.aggregate(method=1)
        agg2 = scores.aggregate(method=2)
        agg3 = scores.aggregate(method=3)
        return (marker_a1, marker, agg1, agg2, agg3)

    # helper functions
    def compute_liwc_reverse_dict(self):
        self.liwc_patterns = {}
        with open(pkg_resources.resource_filename("socialkit",
            "data/coord-liwc-patterns.txt"), "r") as f:
            for line in f:
                cat, pat = line.strip().split("\t")
                self.liwc_patterns[cat] = re.compile(pat, re.IGNORECASE)

    def annot_liwc_cats(self):
        # add liwc_categories field to each utterance
        for k in self.corpus.utterances:
            self.corpus.utterances[k].liwc_categories = set()
        for cat in CoordinationWordCategories:
            pattern = self.liwc_patterns[cat]
            for k, u in self.corpus.utterances.items():
                s = re.search(pattern, u.text)
                if s is not None:
                    self.corpus.utterances[k].liwc_categories.add(cat)

    def scores_over_utterances(self, speakers, utterances,
            speaker_thresh, target_thresh, utterances_thresh,
            speaker_thresh_indiv, target_thresh_indiv, utterances_thresh_indiv,
            utterance_thresh_func=None,
            fine_grained_speakers=False, fine_grained_targets=False):
        assert not isinstance(speakers, str)

        m = self.corpus
        tally = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        cond_tally = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        cond_total = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        n_utterances = defaultdict(lambda: defaultdict(int))
        targets = defaultdict(set)
        for u2 in utterances:
            if u2.reply_to in m.utterances:
                speaker = u2.user if fine_grained_speakers else u2.user.name
                u1 = m.utterances[u2.reply_to]
                target = u1.user if fine_grained_targets else u1.user.name
                if speaker != target:
                    if utterance_thresh_func is None or \
                            utterance_thresh_func(u2, u1):
                        targets[speaker].add(target)
                        n_utterances[speaker][target] += 1
                        for cat in u1.liwc_categories | u2.liwc_categories:
                            if cat in u2.liwc_categories:
                                tally[speaker][cat][target] += 1
                            if cat in u1.liwc_categories:
                                cond_total[speaker][cat][target] += 1
                                if cat in u2.liwc_categories:
                                    cond_tally[speaker][cat][target] += 1
        out = CoordinationScore()
        for speaker in speakers:
            coord_w = {}  # coordination score wrt a category
            for cat in CoordinationWordCategories:
                threshed_cond_total = 0
                threshed_cond_tally = 0
                threshed_tally = 0
                threshed_n_utterances = 0
                for target in targets[speaker]:
                    if tally[speaker][cat][target] >= speaker_thresh_indiv and \
                            cond_total[speaker][cat][target] >= \
                            target_thresh_indiv and \
                            n_utterances[speaker][target] >= \
                            utterances_thresh_indiv:
                        threshed_cond_total += cond_total[speaker][cat][target]
                        threshed_cond_tally += cond_tally[speaker][cat][target]
                        threshed_tally += tally[speaker][cat][target]
                        threshed_n_utterances += n_utterances[speaker][target]
                if threshed_cond_total >= max(target_thresh, 1) and \
                        threshed_tally >= speaker_thresh and \
                        threshed_n_utterances >= max(utterances_thresh, 1):
                    coord_w[cat] = threshed_cond_tally / threshed_cond_total - \
                            threshed_tally / threshed_n_utterances
            if len(coord_w) > 0:
                scores = coord_w.values()
                out[speaker] = coord_w
        return out

