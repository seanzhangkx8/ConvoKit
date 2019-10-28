from convokit.transformer import Transformer
from convokit.model import Corpus

class TextProcessor(Transformer):
    
    def __init__(self, proc_fn, output_field, input_field=None, aux_input={}, input_filter=lambda utt_id,corpus,aux: True, verbosity=0):
        
        self.proc_fn = proc_fn
        self.aux_input = aux_input
        self.input_filter = input_filter
        self.input_field = input_field
        self.output_field = output_field
        self.verbosity = verbosity
        self.multi_outputs = isinstance(output_field, list)
    
    def _print_output(self, i):
        return (self.verbosity > 0) and (i > 0) and (i % self.verbosity == 0)
    
    def transform(self, corpus: Corpus):
        # this could in principle support parallelization...somehow.
        total_utts = len(corpus.utterances)
        # if self.multi_outputs:
        #     for out in self.output_field:
        #         if out not in corpus.features:
        #             corpus.features[out] = {}
        # else:
        #     if self.output_field not in corpus.features:
        #         corpus.features[self.output_field] = {}
        for idx, utt_id in enumerate(corpus.get_utterance_ids()):
            
            if self._print_output(idx):
                print('%03d/%03d utterances processed' % (idx, total_utts))
            if not self.input_filter(utt_id, corpus, self.aux_input): continue
            if self.input_field is None:
                text_entry = corpus.get_utterance(utt_id).text
            elif isinstance(self.input_field, str):
                text_entry = corpus.get_feature(utt_id, self.input_field)
                # corpus.features[self.input_field].get(utt_id, None)
            elif isinstance(self.input_field, list):
                text_entry = {field:corpus.get_feature(utt_id, field) for field in self.input_field}
                # {field:corpus.features[field].get(utt_id, None) for field in self.input_field}
                if sum(x is None for x in text_entry.values()):
                    text_entry = None
            if text_entry is None:
                continue
            if len(self.aux_input) == 0:
                result = self.proc_fn(text_entry)
            else:
                result = self.proc_fn(text_entry, self.aux_input)
            if self.multi_outputs:
                for res, out in zip(result, self.output_field):
                    corpus.set_feature(utt_id, out, res)
                    # corpus.features[out][utt_id] = res
            else:
                corpus.set_feature(utt_id, self.output_field, result)
                # corpus.features[self.output_field][utt_id] = result
 
        return corpus