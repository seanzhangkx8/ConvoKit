from convokit.text_processing import TextProcessor

NP_LABELS = set(['nsubj', 'nsubjpass', 'dobj', 'iobj', 'pobj', 'attr'])


class CensorNouns(TextProcessor):

	def __init__(self, output_field, input_field='parsed', verbosity=0):
		TextProcessor.__init__(self, censor_nouns, 
			output_field=output_field, input_field=input_field, verbosity=verbosity)

def _is_noun_ish(tok):
    return (tok['dep'] in NP_LABELS) or \
        (tok['tag'].startswith('NN') or tok['tag'].startswith('PRP')) \
            or (tok['tag'].endswith('DET')  or tok['tag'].endswith('DT'))

def _get_w_det(tok, sent):

    if tok['tag'].startswith('W'): return tok['tok']
    if len(tok['dn']) == 0: return False
    first_tok = sent['toks'][tok['dn'][0]]
    if first_tok['tag'].startswith('W'): return first_tok['tok']
    return False

def _convert_noun(tok, sent):
    if _is_noun_ish(tok):
        has_w = _get_w_det(tok, sent)
        if has_w:
            return has_w.lower()
        else:
            return 'NN~'
    return tok['tok'].lower()

def censor_nouns(text_entry):
    sents = []
    for raw_sent in text_entry:
        sent = {'rt': raw_sent['rt'], 'toks': []}
        for raw_tok in raw_sent['toks']:
            tok = {k: raw_tok[k] for k in ['dep','dn','tag']}
            if 'up' in raw_tok: tok['up'] = raw_tok['up']
            tok['tok'] = _convert_noun(raw_tok, raw_sent)
            sent['toks'].append(tok)
        sents.append(sent)
    return sents