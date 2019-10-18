from convokit.transformer import Transformer
from convokit.model import Corpus

class TextProcessor(Transformer):
    
    def __init__(self, proc_fn, output_field, aux_input={}, input_field=None, verbosity=0):
        
        self.proc_fn = proc_fn
        self.aux_input = aux_input
        self.input_field = input_field
        self.output_field = output_field
        self.verbosity = verbosity
    
    def _print_output(self, i):
        return (self.verbosity > 0) and (i > 0) and (i % self.verbosity == 0)
    
    def transform(self, corpus: Corpus):
        # this could in principle support parallelization...somehow.
        total_utts = len(corpus.utterances)
        corpus.processed_text[self.output_field] = {}
        if self.output_field not in corpus.processed_text:
            corpus.processed_text[self.output_field] = {}
        for idx, utt_id in enumerate(corpus.get_utterance_ids()):
            if self._print_output(idx):
                print('%03d/%03d utterances processed' % (idx, total_utts))
            if self.input_field is None:
                text_entry = corpus.get_utterance(utt_id).text
            else:
                text_entry = corpus.processed_text[self.input_field][utt_id]
            if len(self.aux_input) == 0:
                corpus.processed_text[self.output_field][utt_id] = self.proc_fn(text_entry)
            else:
                corpus.processed_text[self.output_field][utt_id] = self.proc_fn(text_entry, self.aux_input)
        return corpus