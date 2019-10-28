from .textProcessor import TextProcessor

def use_text(text):
	return text.isalpha() or text[1:].isalpha()

class TextToArcs(TextProcessor):
	
	def __init__(self, output_field, input_field='parsed',
				 use_start=True, root_only=False, follow_deps=('conj',),
				 filter_fn=use_text,input_filter=lambda utt_id,corpus,aux: True,
				  verbosity=0):
		aux_input = {'root_only': root_only, 'use_start': use_start, 'follow_deps': follow_deps, 'filter_fn': filter_fn}
		TextProcessor.__init__(self, proc_fn=self._get_arcs_per_message_wrapper, output_field=output_field, input_field=input_field, aux_input=aux_input, input_filter=input_filter, verbosity=verbosity)
	
	
	def _get_arcs_per_message_wrapper(self, text_entry, aux_input={}):
		return get_arcs_per_message(text_entry, aux_input['use_start'], 
			aux_input['root_only'], aux_input['follow_deps'],
			aux_input['filter_fn'])



def _get_arcs_at_root(root, sent, use_start=True, root_only=False, follow_deps=('conj',), filter_fn=use_text):
	
	arcs = set()
	root_tok = root['tok'].lower()
	if not filter_fn(root_tok): return arcs
	arcs.add(root_tok + '_*')
	
	next_elems = []
	for kid_idx in root['dn']:
		kid = sent['toks'][kid_idx]
		if kid['dep'] in ['cc']: continue
		
		kid_tok = kid['tok'].lower()
		if filter_fn(kid_tok):
			if kid['dep'] not in follow_deps:
				arcs.add(root_tok + '_' + kid_tok)
			if (not root_only) or (kid['dep'] in follow_deps):
				next_elems.append(kid)
	if use_start:
		first_elem = sent['toks'][0]
		first_tok = first_elem['tok'].lower()
		if filter_fn(first_tok): 
			arcs.add(first_tok + '>*')
			if (1 not in first_elem['dn']) and (len(sent['toks']) >= 2):
				second_elem = sent['toks'][1]
				if 0 not in second_elem['dn']:
					second_tok = second_elem['tok'].lower()
					if filter_fn(second_tok): arcs.add(first_tok + '>' + second_tok)
	for next_elem in next_elems:
		arcs.update(_get_arcs_at_root(next_elem, sent, 
				use_start=False, root_only=root_only, follow_deps=follow_deps, filter_fn=filter_fn))
	return arcs

def get_arcs_per_message(message, use_start=True, root_only=False, follow_deps=('conj',), filter_fn=use_text):
	return [' '.join(sorted(_get_arcs_at_root(sent['toks'][sent['rt']], sent, use_start=use_start, root_only=root_only, follow_deps=follow_deps, filter_fn=filter_fn)))
		for sent in message]