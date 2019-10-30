
from .textProcessor import TextProcessor

class TokensToString(TextProcessor):
	
	def __init__(self, output_field, input_field='parsed',
				 token_formatter=lambda x: x['tok'], token_filter=lambda x: True, 
				  input_filter=None, verbosity=0):
		aux_input = {'token_formatter': token_formatter, 'token_filter': token_filter}
		TextProcessor.__init__(self, proc_fn=self._get_token_string_wrapper, output_field=output_field, 
							  input_field=input_field, aux_input=aux_input, 
							  input_filter=input_filter, verbosity=verbosity)
	
	
	def _get_token_string_wrapper(self, text_entry, aux_input={}):
		return self._get_token_string(text_entry, aux_input['token_formatter'], aux_input['token_filter'])

	def _get_token_string(self, parse, token_formatter=lambda x: x['tok'], token_filter=lambda x: True):
		return [' '.join(token_formatter(tok) for tok in sent['toks'] if token_filter(tok)) for sent in parse]
