import nltk
import spacy

from .textProcessor import TextProcessor

class TextParser(TextProcessor):
	
	def __init__(self, output_field, mode='parse', input_field=None, spacy_nlp=None, sent_tokenizer=None, verbosity=0):

		self.mode = mode
		aux_input = {'mode': mode}

		if spacy_nlp is None:
			try:
				if mode == 'parse':
					aux_input['spacy_nlp'] = spacy.load('en', disable=['ner'])
				elif mode == 'tag':
					aux_input['spacy_nlp'] = spacy.load('en', disable=['ner','parser'])
				elif mode == 'tokenize':
					aux_input['spacy_nlp'] = spacy.load('en', disable=['ner','parser', 'tagger'])
			except OSError:
				print("Convokit requires a SpaCy English model to be installed. Run `python -m spacy download en` and retry.")
				sys.exit()
		else:
			aux_input['spacy_nlp'] = spacy_nlp

		if mode in ('tag','tokenize'):
			if sent_tokenizer is None:
				try:
					aux_input['sent_tokenizer'] = nltk.data.load('tokenizers/punkt/english.pickle')
				except OSError:
					print("Convokit requires nltk data to be downloaded. Run `python -m nltk.downloader all` and retry.")
					sys.exit()
			else:
				aux_input['sent_tokenizer'] = sent_tokenizer

		TextProcessor.__init__(self, self._process_text_wrapper, output_field, aux_input, input_field, verbosity)
	
	def _process_text_wrapper(self, text, aux_input={}):
		return process_text(text, aux_input.get('mode','parse'), 
						aux_input.get('sent_tokenizer',None), aux_input.get('spacy_nlp',None))

# these could in principle live in a separate text_utils.py file.
def _process_token(token_obj, mode='parse', offset=0):
	if mode == 'tokenize':
		token_info = {'tok': token_obj.text}
	else:
		token_info = {'tok': token_obj.text, 'tag': token_obj.tag_}
	if mode == 'parse':
		token_info['dep'] = token_obj.dep_
		if token_info['dep'] != 'ROOT':
			token_info['up'] = next(token_obj.ancestors).i - offset
		token_info['dn'] = [x.i - offset for x in token_obj.children]
	return token_info

def _process_sentence(sent_obj, mode='parse', offset=0):
	tokens = []
	for token_obj in sent_obj:
		tokens.append(_process_token(token_obj, mode, offset))
	if mode == 'parse':
		return {'rt': sent_obj.root.i - offset, 'toks': tokens}
	else:
		return {'toks': tokens}

def process_text(text, mode='parse', sent_tokenizer=None, spacy_nlp=None):
	
	if spacy_nlp is None:
		raise ValueError('spacy object required')
	if mode in ('tag', 'tokenize'):
		if sent_tokenizer is None:
			raise ValueError('sentence tokenizer required')
	
	if mode == 'parse':
		sents = spacy_nlp(text).sents
	else:
		sents = [spacy_nlp(x) for x in sent_tokenizer.tokenize(text)]
	
	sentences = []
	offset = 0
	for sent in sents:
		curr_sent = _process_sentence(sent, mode, offset)
		sentences.append(curr_sent)
		offset += len(curr_sent['toks'])
	return sentences


