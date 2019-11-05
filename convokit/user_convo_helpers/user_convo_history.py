from convokit.transformer import Transformer
from convokit.model import Corpus

from collections import defaultdict

class UserConvoHistory(Transformer):

    '''
        For each user, pre-computes a list of all of their utterances, organized by the conversation they participated in. Annotates user with the following:
            * `n_convos`: number of conversations
            * `start_time`: time of first utterance, across all conversations
            * `conversations`: a dictionary keyed by conversation id, where entries consist of:
                * `idx`: the index of the conversation, in terms of the time of the first utterance contributed by that particular user (i.e., `idx=0` means this is the first conversation the user ever participated in)
                * `n_utterances`: the number of utterances the user contributed in the conversation
                * `start_time`: the timestamp of the user's first utterance in the conversation
                * `utterance_ids`: a list of ids of utterances contributed by the user, ordered by timestamp.
        In case timestamps are not provided with utterances, the present behavior is to sort just by utterance id. 
        
        :param utterance_filter: function that returns True for an utterance that counts towards a user having participated in that conversation. (e.g., one could filter out conversations where the user contributed less than k words per utterance)
    '''
    
    def __init__(self, utterance_filter=None):
        if utterance_filter is None:
            self.utterance_filter = lambda x: True
        else:
            self.utterance_filter = utterance_filter
    
    def transform(self, corpus: Corpus):
        user_to_convo_utts = defaultdict(lambda: defaultdict(list))
        for utterance in corpus.iter_utterances():
            if not self.utterance_filter(utterance): continue
            user_to_convo_utts[utterance.user.name][utterance.root].append(
                (utterance.id, utterance.timestamp))
        for user, convo_utts in user_to_convo_utts.items():
            user_convos = {}
            for convo, utts in convo_utts.items():
                sorted_utts = sorted(utts, key=lambda x: (x[1], x[0]))
                user_convos[convo] = {'utterance_ids': [x[0] for x in sorted_utts], 
                                    'start_time': sorted_utts[0][1],
                                     'n_utterances': len(sorted_utts)}
            corpus.get_user(user).add_meta('conversations', user_convos)
        
        for user in corpus.iter_users():
            if 'conversations' not in user.meta: continue
            user.add_meta('n_convos', len(user.meta['conversations']))
            
            sorted_convos = sorted(user.meta['conversations'].items(), key=lambda x: (x[1]['start_time'], x[1]['utterance_ids'][0]))
            user.add_meta('start_time', sorted_convos[0][1]['start_time'])
            for idx, (convo_id, _) in enumerate(sorted_convos):
                user.meta['conversations'][convo_id]['idx'] = idx
        return corpus