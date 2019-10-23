from convokit import Corpus
from convokit.text_processing import TextProcessor, TextParser, TokensToString, TextToArcs
from convokit.phrasing_motifs import CensorNouns, QuestionSentences, PhrasingMotifs

import os
import sys

VERBOSITY = 1000

def preprocess_text(text): # replace w your own
    text = text.replace(' -- ', ' ')
    return text

if __name__ == '__main__':

	ROOT_DIR = sys.argv[1]

	print('reading corpus', ROOT_DIR)
	corpus = Corpus(ROOT_DIR)

	print('preprocessing text')
	text_prep = TextProcessor(preprocess_text, 'text', verbosity=VERBOSITY)
	corpus = text_prep.transform(corpus)

	print('parsing text')
	textparser = TextParser('parsed', input_field='text', verbosity=VERBOSITY)
	corpus = textparser.transform(corpus)

	# corpus.load_processed_text(['parsed'])

	print('converting text to simple serializable form')
	tok_to_str = TokensToString('tok_str', verbosity=VERBOSITY)
	corpus = tok_to_str.transform(corpus)

	print('extracting full arcs')
	text_to_arc = TextToArcs('arcs', verbosity=VERBOSITY)
	corpus = text_to_arc.transform(corpus)

	print('censoring nouns')
	censor_nouns = CensorNouns('parsed_censored', verbosity=VERBOSITY)
	corpus = censor_nouns.transform(corpus)

	print('extracting noun-censored root arcs')
	text_to_arc_mini_censored = TextToArcs('arcs_censored', input_field='parsed_censored', root_only=True, verbosity=VERBOSITY)
	corpus = text_to_arc_mini_censored.transform(corpus)


	# corpus.load_processed_text(['arcs_censored', 'tok_str'])

	print('extracting question-only arcs')
	qs = QuestionSentences(output_field='question_arcs', input_field='arcs_censored', use_caps=True)
	corpus = qs.transform(corpus)


	print('extracting phrasing motifs')
	try:
		os.mkdir(os.path.join(ROOT_DIR, 'pm_model'))
	except: pass
	pm_model = PhrasingMotifs('question_motifs','question_arcs',50,verbosity=20000)
	pm_model.fit(corpus)
	pm_model.print_top_phrasings(25)
	corpus = pm_model.transform(corpus)
	pm_model.dump_model(os.path.join(ROOT_DIR, 'pm_model'))

	print('dumping corpus to', corpus.original_corpus_path)
	corpus.dump_processed_text(['parsed', 'tok_str', 'arcs', 'arcs_censored', 'question_arcs', 'question_motifs'])
	# corpus.dump_processed_text(['question_arcs','question_motifs'])