from .textProcessor import TextProcessor

def use_text(tok, sent):
	return tok['tok'].isalpha() or tok['tok'][1:].isalpha()

class TextToArcs(TextProcessor):
	
	def __init__(self, output_field, input_field='parsed',
				 use_start=True, root_only=False, follow_deps=('conj',),
				 filter_fn=use_text,input_filter=None,
				  verbosity=0):
		aux_input = {'root_only': root_only, 'use_start': use_start, 'follow_deps': follow_deps, 'filter_fn': filter_fn}
		TextProcessor.__init__(self, proc_fn=self._get_arcs_per_message_wrapper, output_field=output_field, input_field=input_field, aux_input=aux_input, input_filter=input_filter, verbosity=verbosity)
	
	
	def _get_arcs_per_message_wrapper(self, text_entry, aux_input={}):
		return get_arcs_per_message(text_entry, aux_input['use_start'], 
			aux_input['root_only'], aux_input['follow_deps'],
			aux_input['filter_fn'])



def _get_arcs_at_root(root, sent, use_start=True, root_only=False, follow_deps=('conj',), filter_fn=use_text):
	
	arcs = set()
	if not filter_fn(root, sent): return arcs
	arcs.add(root['tok'].lower() + '_*')
	
	next_elems = []
	for kid_idx in root['dn']:
		kid = sent['toks'][kid_idx]
		if kid['dep'] in ['cc']: continue
		
		if filter_fn(kid, sent):
			if (kid['dep'] not in follow_deps) and (root['tok'].lower() != kid['tok'].lower()):
				arcs.add(root['tok'].lower() + '_' + kid['tok'].lower())
			if (not root_only) or (kid['dep'] in follow_deps):
				next_elems.append(kid)
	if use_start:
		first_elem = sent['toks'][0]
		if filter_fn(first_elem, sent): 
			arcs.add(first_elem['tok'].lower() + '>*')
			if (1 not in first_elem['dn']) and (len(sent['toks']) >= 2):
				second_elem = sent['toks'][1]
				if 0 not in second_elem['dn']:
					if filter_fn(second_elem, sent) and (first_elem['tok'].lower() != second_elem['tok'].lower()): arcs.add(first_elem['tok'].lower() + '>' + second_elem['tok'].lower())
	for next_elem in next_elems:
		arcs.update(_get_arcs_at_root(next_elem, sent, 
				use_start=False, root_only=root_only, follow_deps=follow_deps, filter_fn=filter_fn))
	return arcs

def get_arcs_per_message(message, use_start=True, root_only=False, follow_deps=('conj',), filter_fn=use_text):
	return [' '.join(sorted(_get_arcs_at_root(sent['toks'][sent['rt']], sent, use_start=use_start, root_only=root_only, 
                                              follow_deps=follow_deps, filter_fn=filter_fn)))
		for sent in message]