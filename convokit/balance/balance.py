from convokit.model import Corpus, Speaker, Conversation
from convokit.transformer import Transformer
from collections import Counter
from scipy.stats import wilcoxon, mannwhitneyu
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import re
import random

class Balance(Transformer):
    def __init__(
        self,
        primary_threshold=0.50001,
        window_ps_threshold=0.6,
        window_size=2.5,
        sliding_size=30,
        cur_cut=0,
        remove_first_last_utt=False,
        convo_filter=lambda convo: True
    ):
        self.primary_threshold = primary_threshold
        self.window_ps_threshold = window_ps_threshold
        self.window_size = window_size
        self.sliding_size = sliding_size
        self.cur_cut = cur_cut
        self.remove_first_last_utt = remove_first_last_utt
        self.convo_filter = convo_filter
        
    def transform(
        self,
        corpus: Corpus
    ):
        # annotate utt.meta['utt_group'] based on sp.meta['group'] is groupA or groupB
        if 'utt_group' not in corpus.random_utterance().meta:
            for utt in tqdm(corpus.iter_utterances(), desc="Annotating utterance group: "):
                utt.meta['utt_group'] = utt.get_conversation().meta['speaker_group'][utt.speaker.id]

        # Now assume groupA and groupB are annotated for each utterance
        for convo in tqdm(corpus.iter_conversations(selector=self.convo_filter), desc="Annotating conversation Balance Info: "):
            primary_speaker = self._get_ps(corpus, convo.get_utterance_ids())
            convo.meta['primary_speaker'] = primary_speaker
            convo.meta['secondary_speaker'] = 'groupA' if primary_speaker == 'groupB' else 'groupB'

            balance = self._convo_balance_score(corpus, convo.id)
            convo.meta['balance_score'] = balance if balance != -1 else None
            balance_lst = self._convo_balance_lst(corpus, convo.id)
            convo.meta['balance_lst'], convo.meta['no_speaking_time_count'], convo.meta['all_window_count'] = balance_lst


    def _tokenize(self, text):
        text = text.lower()
        text = re.findall('[a-z]+', text)
        return text

    def _longer_than_xwords(self, corpus, utt_id, x=None):
        if x is None:
            x = self.cur_cut
        utt = corpus.get_utterance(utt_id)
        return len(self._tokenize(utt.text)) >= x
    
    def _rhythm_count_utt_time(self, corpus, utt_lst, cur_cut=None):
        if cur_cut is None:
            cur_cut = self.cur_cut
        valid_utt = [utt_id for utt_id in utt_lst if self._longer_than_xwords(corpus, utt_id)]
        if len(valid_utt) == 0: return 0, 0
        time_A = 0
        time_B = 0
        for utt_id in valid_utt:
            utt = corpus.get_utterance(utt_id)
            time = utt.meta['stop'] - utt.meta['start']
            if utt.meta['utt_group'] == 'groupA':
                time_A += time
            elif utt.meta['utt_group'] == 'groupB':
                time_B += time
        return time_A, time_B

    def _get_ps(self, corpus, utt_lst, primary_threshold=None):
        if primary_threshold is None:
            primary_threshold = self.primary_threshold
        assert primary_threshold > 0.5, "Primary Threshold should greater than 0.5"
        if len(utt_lst) == 0: return None
        time_A, time_B = self._rhythm_count_utt_time(corpus, utt_lst)
        total_speaking_time = time_A + time_B
        if time_A > (total_speaking_time * primary_threshold):
            return 'groupA'
        elif time_B > (total_speaking_time * primary_threshold):
            return 'groupB'
        else:
            return None
        
    def _sliding_window(self, corpus, convo_id, window_size=None, sliding_size=None):
        if window_size is None:
            window_size = self.window_size
        if sliding_size is None:
            sliding_size = self.sliding_size
        convo = corpus.get_conversation(convo_id)
        if self.remove_first_last_utt:
            utt_lst = convo.get_utterance_ids()[1:-1]
        else:
            utt_lst = convo.get_utterance_ids()

        cur_start_time = corpus.get_utterance(utt_lst[0]).meta['start']
        cur_end_time = cur_start_time + (window_size * 60)
        prev_window_last_utt_id = utt_lst[0]
        convo_end_time = corpus.get_utterance(utt_lst[-1]).meta['stop']

        all_windows = []

        while prev_window_last_utt_id != utt_lst[-1] and cur_end_time < convo_end_time:
            cur_window_groupA_speaking_time = 0
            cur_window_groupB_speaking_time = 0

            for i, utt_id in enumerate(utt_lst):
                utt = corpus.get_utterance(utt_id)
                # case 1: utterances in previous windows
                if utt.meta['stop'] < cur_start_time: continue

                # case 2: last utt of the window
                if utt.meta['stop'] > cur_end_time:
                    # the entire utt not in the window
                    if utt.meta['start'] > cur_end_time:
                        prev_window_last_utt_id = utt_lst[i-1]
                    # part of the utt in the window
                    else:
                        if utt.meta['utt_group'] == 'groupA':
                            cur_window_groupA_speaking_time += cur_end_time - utt.meta['start']
                        elif utt.meta['utt_group'] == 'groupB':
                            cur_window_groupB_speaking_time += cur_end_time - utt.meta['start']
                        prev_window_last_utt_id = utt_id
                    # put window data in all_windows only at the terminating point: last utt of the window
                    all_windows.append({'groupA' : cur_window_groupA_speaking_time, 'groupB' : cur_window_groupB_speaking_time})
                    break
                
                # case 3: first utt of the window
                if i == 0:
                    # entire utt not in window
                    if utt.meta['stop'] < cur_start_time:
                        continue
                    # part of the utt in window
                    elif utt.meta['start'] < cur_start_time and utt.meta['stop'] > utt.meta['start']:
                        if utt.meta['utt_group'] == 'groupA':
                            cur_window_groupA_speaking_time += utt.meta['stop'] - cur_start_time
                        elif utt.meta['utt_group'] == 'groupB':
                            cur_window_groupB_speaking_time += utt.meta['stop'] - cur_start_time
                    # entire utt in window
                    else:
                        if utt.meta['utt_group'] == 'groupA':
                            cur_window_groupA_speaking_time += utt.meta['stop'] - utt.meta['start']
                        elif utt.meta['utt_group'] == 'groupB':
                            cur_window_groupB_speaking_time += utt.meta['stop'] - utt.meta['start']

                # case 4: utt in middle of the window
                else:
                    if utt.meta['utt_group'] == 'groupA':
                        cur_window_groupA_speaking_time += utt.meta['stop'] - utt.meta['start']
                    elif utt.meta['utt_group'] == 'groupB':
                        cur_window_groupB_speaking_time += utt.meta['stop'] - utt.meta['start']

            # update window start end time
            cur_start_time += sliding_size
            cur_end_time += sliding_size

        return all_windows
    
    def _convo_balance_score(self, corpus, convo_id):
        """
        Annotate with overall time-based measurement.
        """
        if self.remove_first_last_utt:
            utt_lst = corpus.get_conversation(convo_id).get_utterance_ids()[1:-1]
        else:
            utt_lst = corpus.get_conversation(convo_id).get_utterance_ids()
        timeA, timeB = self._rhythm_count_utt_time(corpus, utt_lst)
        total_time = timeA + timeB
        if total_time == 0:
            print(convo_id)
            return -100
        return timeA / total_time if timeA >= timeB else timeB / total_time

    def _convo_balance_lst(self, corpus, convo_id):
        """
        Annotate individual_conversation_floor_lst
        """
        groups = self._sliding_window(corpus, convo_id)
        balance_lst = []
        no_speaking_time_count = 0
        all_window_count = 0
        for window in groups:
            all_window_count += 1
            window_ps_time = window['groupA']
            window_ss_time = window['groupB']
            window_total_time = window_ps_time + window_ss_time
            window_id = 0
            if window_total_time == 0: # No Speaking Time in the window
                window_id = -100
                no_speaking_time_count += 1
            elif window_ps_time >= window_ss_time:
                window_id = window_ps_time / window_total_time if window_ps_time / window_total_time > self.window_ps_threshold else 0
            elif window_ps_time < window_ss_time:
                window_id = -1 * window_ss_time / window_total_time if window_ss_time / window_total_time > self.window_ps_threshold else 0

            if window_id == 0:
                balance_lst.append(0)
            elif window_id > 0:
                balance_lst.append(1)
            else:
                balance_lst.append(-1)
        return balance_lst, no_speaking_time_count, all_window_count
