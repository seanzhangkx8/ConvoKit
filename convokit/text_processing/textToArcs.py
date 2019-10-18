from convokit.transformer import Transformer
from convokit.model import Corpus
from .textProcessor import TextProcessor

# this is admittedly a bit ad-hoc

NP_LABELS = set(['nsubj', 'nsubjpass', 'dobj', 'iobj', 'pobj', 'attr'])
CONJ_LABELS = set(['conj'])

class TextToArcs(TextProcessor):
	
	def __init__(self, output_field, 
				 use_start=True, censor_nouns=False, root_only=False, 
				 input_field='parsed', verbosity=0):
		aux_input = {'censor_nouns': censor_nouns, 'root_only': root_only, 'use_start': use_start}
		TextProcessor.__init__(self, self._get_arcs_per_message_wrapper, output_field, aux_input,
							  input_field, verbosity)
	
	
	def _get_arcs_per_message_wrapper(self, text_entry, aux_input={}):
		return self._get_arcs_per_message(text_entry, aux_input['use_start'], aux_input['censor_nouns'], aux_input['root_only'])

	def _is_noun_ish(self, tok):
		return (tok['dep'] in NP_LABELS) or \
			(tok['tag'].startswith('NN') or tok['tag'].startswith('PRP')) \
				or (tok['tag'].endswith('DET')  or tok['tag'].endswith('DT'))

	def _get_w_det(self, tok, sent):

		if tok['tag'].startswith('W'): return tok['tok']
		if len(tok['dn']) == 0: return False
		first_tok = sent['toks'][tok['dn'][0]]
		if first_tok['tag'].startswith('W'): return first_tok['tok']
		return False

	def _get_tok_text(self, tok, sent, censor_nouns=False):
		if self._is_noun_ish(tok) and censor_nouns:
			has_w = self._get_w_det(tok, sent)
			if has_w:
				return has_w.lower()
			else:
				return 'NN~'
		elif tok['tok'][0].isalpha():
			return tok['tok'].lower()
		elif tok['tok'].isdigit():
			return 'NUM~'
		elif tok['tok'] == '?':
			return '?'
		else:
			return tok['tok'][1:].lower()

	def _is_usable(self, text):
		return text.isalpha() or text[1:].isalpha() or (text == '?')

	def _get_arcs_at_root(self, root, sent, use_start=True, censor_nouns=False, root_only=False):
		
		arcs = set()
		root_tok = self._get_tok_text(root, sent, censor_nouns)
		if not self._is_usable(root_tok): return arcs
		arcs.add(root_tok + '_*')
		
		next_elems = []
		for kid_idx in root['dn']:
			kid = sent['toks'][kid_idx]
			if kid['dep'] in ['cc']: continue
			
			kid_tok = self._get_tok_text(kid, sent, censor_nouns)
			if self._is_usable(kid_tok):
				if kid['dep'] not in CONJ_LABELS:
					arcs.add(root_tok + '_' + kid_tok)
				if (not root_only) or (kid['dep'] in CONJ_LABELS):
					next_elems.append(kid)
		if use_start:
			first_elem = sent['toks'][0]
			first_tok = self._get_tok_text(first_elem, sent, censor_nouns)
			if self._is_usable(first_tok): 
				if (1 not in first_elem['dn']) and (len(sent['toks']) >= 2):
					second_elem = sent['toks'][1]
					if 0 not in second_elem['dn']:
						second_tok = self._get_tok_text(second_elem, sent, censor_nouns)
						if self._is_usable(second_tok): arcs.add(first_tok + '>' + second_tok)
		for next_elem in next_elems:
			arcs.update(self._get_arcs_at_root(next_elem, sent, 
					use_start=False, censor_nouns=censor_nouns, root_only=root_only))
		return arcs

	def _get_arcs_per_message(self, message, use_start=True, censor_nouns=False, root_only=False):
		return [sorted(self._get_arcs_at_root(sent['toks'][sent['rt']], sent, use_start=use_start, censor_nouns=censor_nouns, root_only=root_only))
			for sent in message]