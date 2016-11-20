"""Coordination features
(https://www.cs.cornell.edu/~cristian/Echoes_of_power.html)."""

import pkg_resources
import re
from collections import defaultdict

CoordinationWordCategories = ["article", "auxverb", "conj", "adverb",
        "ppron", "ipron", "preps", "quant"]

class Coordination:
    """Encapsulates computation of coordination-based features for a particular
    model.

    Args:
        model (Model): the model to compute features for.

    Attributes:
        model (Model): the model associated with the coordination object.
    """

    def __init__(self, model):
        self.model = model
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

        Args:
            speakers: A collection of usernames or user objects corresponding to
                the speakers we want to compute scores for. Can also be a single
                username/user object if only one speaker. If a user object is
                passed in, the scoring will count users as unique if they have
                different user infos, which can be used to compare the same user
                across different attributes.
            group: A collection of usernames or user objects corresponding to
                the group of targets. Can also be a single username/user object
                if only one target.
            speaker_thresh (int, optional, default=0): Thresholds based on
                minimum number of times the speaker uses each coordination
                marker.
            target_thresh (int, optional, default=3): Thresholds based on
                minimum number of times the target uses each coordination
                marker.
            utterances_thresh (int, optional, default=0): Thresholds based on
                the minimum number of utterances for each speaker.
            speaker_thresh_indiv (int, optional, default=0): Like
                `speaker_thresh` but only considers the utterances
                between a speaker and a single target; thresholds whether the
                utterances for a single target should be considered for a
                particular speaker.
            target_thresh_indiv (int, optional, default=0): Like
                `target_thresh` but thresholds whether a single target's
                utterances should be considered for a particular speaker.
            utterances_thresh_indiv (int, optional, default=0): Like
                `utterances_thresh` but thresholds whether a single target's
                utterances should be considered for a particular speaker.
            utterance_thresh_func (function, optional): Optional utterance-level
                threshold function that takes in a speaker `Utterance` and
                the `Utterance` the speaker replied to, and returns a `bool`
                corresponding to whether or not to include the utterance in
                scoring.

        Returns:
            A dictionary of scores:

            {
                speaker_1: { dictionary of scores by coordination marker },
                speaker_2: scores,
                ...
            }

            The keys are of the same types as the input: if a username was 
                passed in, the corresponding key will be a username, etc.
        """
        if isinstance(speakers, str): speakers = [speakers]
        if isinstance(group, str): group = [group]

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
        for u in self.model.utterances.values():
            speaker = u.user if fine_grained_speakers else u.user.name
            if speaker in speakers:
                if u.reply_to is not None:
                    reply_to = self.model.utterances[u.reply_to]
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
        
        Args:
            pairs (Collection): collection of (speaker, target) pairs where
                each speaker and target can be either a username or a user
                object.
            
            Also accepted: all threshold arguments accepted by `score`.

        Returns:
            Dictionary of scores indexed by (speaker, target) pairs.

            Each value is itself a dictionary with scores indexed by
                coordination marker.
        """
        self.precompute()
        all_scores = {}
        for (speaker, target), utterances in pairs.items():
            scores = self.scores_over_utterances([speaker], utterances,
                    speaker_thresh, target_thresh, utterances_thresh,
                    speaker_thresh_indiv, target_thresh_indiv,
                    utterances_thresh_indiv, utterance_thresh_func,
                    not isinstance(speaker, str), not isinstance(target, str))
            for m in scores.values():
                all_scores[speaker, target] = m
        return all_scores

    def score_report(self, all_scores):
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

        Args:
            scores (dict): Scores to produce a report for.

        Returns:
            A tuple (marker_a1, marker, agg1, agg2, agg3):
            - marker_a1 is a dictionary of aggregate scores by marker,
                using the scores only over users included in Aggregate 1.
            - marker is a dictionary of aggregate scores by marker,
                using the scores of all users with a coordination score for
                that marker.
            - agg1, agg2 and agg3 are Aggregate 1, 2 and 3 scores respectively.
        """

        a1_scores_by_marker = defaultdict(list)
        scores_by_marker = defaultdict(list)
        for speaker, scores in all_scores.items():
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
        for speaker, scoredict in all_scores.items():
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
        return a1_avg_by_marker, avg_by_marker, agg1, agg2, agg3

    # helper functions
    def get_liwcpattern(self, feature):
        liwc_patterns1=""
        for k in self.liwc_reversed[feature]:
            if not k.endswith("*"):
                liwc_patterns1+="\\b"+k+"\\b|"
            else:
                liwc_patterns1+="\\b"+k.strip("*")+"|"
        liwc_patterns1=liwc_patterns1[:-1]
        return re.compile(liwc_patterns1,re.I)

    def compute_liwc_reverse_dict(self):
        '''
        Created on Jan 19, 2012

        @author: cristian
        '''

        lkey={}
        liwc={}
        f=open(pkg_resources.resource_filename("socialkit", "data/LIWC.dic"))
                #'data/LIWC2007_English080730.dic')
        lines=f.readlines()
        f.close()

        lkey2={}


        lc=1
        l=lines[lc].strip()
        while l != "%":
            (k,v)=l.split()
            lkey2[int(k)]=v.lower()
            lc+=1
            l=lines[lc].strip()


        lc+=1
        while lc<len(lines):
            l=lines[lc].strip()
            entry=l.split()
            for k in entry[1:]:
                try: 
                    liwc.setdefault(entry[0],[]).append(lkey2[int(k)])
                except ValueError:
                    pass
                    #print entry[0],k,"not an integer"
            lc+=1

        liwc_nostar_toint={} # indexed by lemmas (no "*"s)
        entry_to_int={}
        int_to_entry={}
        ec=0
        all_entries=list(set(lkey.values()).union(set(lkey2.values())))

        for v in sorted(all_entries):
            entry_to_int[v]=ec
            int_to_entry[ec]=v
            ec+=1

        for k in liwc:
            liwc_nostar_toint.setdefault(k.strip("*"),[]).extend(
                    [entry_to_int[v] for v in liwc[k]])
        self.liwc_reversed = {}
        for w in liwc:
            for e in liwc[w]:
                self.liwc_reversed.setdefault(e,[]).append(w)

    def annot_liwc_cats(self):
        # add liwc_categories field to each utterance
        for k in self.model.utterances:
            self.model.utterances[k].liwc_categories = set()
        for cat in CoordinationWordCategories:
            pattern = self.get_liwcpattern(cat)
            for k, u in self.model.utterances.items():
                s = re.search(pattern, u.text)
                if s is not None:
                    self.model.utterances[k].liwc_categories.add(cat)

    def scores_over_utterances(self, speakers, utterances,
            speaker_thresh, target_thresh, utterances_thresh,
            speaker_thresh_indiv, target_thresh_indiv, utterances_thresh_indiv,
            utterance_thresh_func=None,
            fine_grained_speakers=False, fine_grained_targets=False):
        assert not isinstance(speakers, str)

        m = self.model
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
        out = {}
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

