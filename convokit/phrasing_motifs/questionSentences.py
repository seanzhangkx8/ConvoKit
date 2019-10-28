from convokit.text_processing import TextProcessor

class QuestionSentences(TextProcessor):
    
    def __init__(self, output_field, input_field, use_caps=True, filter_field='parsed',
        input_filter=lambda utt_id, corpus, aux: True,
        verbosity=0):


        aux_input = {'input_field': input_field,
        'filter_field': filter_field, 'use_caps': use_caps}


        TextProcessor.__init__(self, self._get_question_sentences, output_field=output_field, input_field=[input_field, filter_field], input_filter=input_filter, aux_input=aux_input, verbosity=verbosity)
    

    def _get_question_sentences(self, text_entry, aux_input):

        text = text_entry[aux_input['input_field']]
        parse = text_entry[aux_input['filter_field']]
        sents = []
        for input_sent, filter_sent in zip(text, parse):
            if isinstance(filter_sent, dict):
                if filter_sent['toks'][-1]['tok'] != '?': 
                    continue
                if aux_input['use_caps'] and not filter_sent['toks'][0]['tok'][0].isupper(): 
                    continue
            else:
                if '?' not in filter_sent: 
                    continue
                if aux_input['use_caps'] and not filter_sent[0].isupper():
                    continue
            sents.append(input_sent)
        return sents

    # def transform(self, corpus):
    #     corpus.features[self.output_field] = {}
    #     for utt_id in corpus.get_utterance_ids():
    #         if corpus.get_utterance(utt_id).meta['is_question']:
    #             corpus.features[self.output_field][utt_id] = [input_sent
    #                         for filter_sent, input_sent in zip(corpus.get_feature(utt_id, self.filter_field).split('\n'),
    #               corpus.get_feature(utt_id, self.input_field)) 
    #             if (filter_sent[-1]['tok'] == '?')
    #             and ((not self.use_caps) or (filter_sent[0].isupper()))]
    #         else:
    #             corpus.features[self.output_field][utt_id] = []
    #     return corpus