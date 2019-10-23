from convokit.transformer import Transformer

class QuestionSentences(Transformer):
    
    def __init__(self, output_field, input_field, use_caps=True, filter_field='tok_str'):
        self.output_field = output_field
        self.input_field = input_field
        self.use_caps = use_caps
        self.filter_field = filter_field
    
    def transform(self, corpus):
        corpus.processed_text[self.output_field] = {}
        for utt_id in corpus.get_utterance_ids():
            if corpus.get_utterance(utt_id).meta['is_question']:
                corpus.processed_text[self.output_field][utt_id] = [input_sent
                            for filter_sent, input_sent in zip(corpus.get_processed_text(utt_id, self.filter_field).split('\n'),
                  corpus.get_processed_text(utt_id, self.input_field)) 
                if ('?' in filter_sent)
                and ((not self.use_caps) or (filter_sent[0].isupper()))]
            else:
                corpus.processed_text[self.output_field][utt_id] = []
        return corpus