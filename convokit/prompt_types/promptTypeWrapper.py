from convokit import Corpus
from convokit.text_processing import TextProcessor, TextParser, TextToArcs
from convokit.phrasing_motifs import CensorNouns, QuestionSentences, PhrasingMotifs
from convokit.prompt_types import PromptTypes
from convokit.convokitPipeline import ConvokitPipeline
from convokit.transformer import Transformer
from convokit.model import Utterance

import os

class PromptTypeWrapper(Transformer):
	
	def __init__(self, output_field='prompt_types', n_types=8,
				questions_only=True, enforce_caps=True, recompute_all=False, 
				 min_support=100, svd__n_components=25, prompt__tfidf_max_df=.1,ref__tfidf_max_df=.1,
					max_dist=.9,
				 random_state=None, verbosity=10000, 
				):

		self.random_state=random_state
		pipe = [
			('parser', TextParser(verbosity=verbosity, 
				 input_filter=lambda utt, aux: recompute_all or (utt.get_info('parsed') is None))),
			('censor_nouns', CensorNouns('parsed_censored', 
				 input_filter=lambda utt, aux: recompute_all or (utt.get_info('parsed_censored') is None),
										 verbosity=verbosity)),
			('shallow_arcs', TextToArcs('arcs_censored', input_field='parsed_censored',
				input_filter=lambda utt, aux: recompute_all or (utt.get_info('arcs_censored') is None),
									   root_only=True, verbosity=verbosity))
			
		]
		
		if questions_only:
			pipe.append(
				('question_sentence_filter', QuestionSentences('question_arcs',
									input_field='arcs_censored', 
								   input_filter=lambda utt, aux: recompute_all or utt.meta['is_question'],
									use_caps=enforce_caps, verbosity=verbosity))
			)
		
			prompt_input_field = 'question_arcs'
			prompt_filter = lambda utt, aux: utt.meta['is_question']
			ref_filter = lambda utt, aux: (not utt.meta['is_question']) and (utt.reply_to is not None)
		else:
			prompt_input_field = 'arcs_censored'
			prompt_filter = lambda utt, aux: True
			ref_filter = lambda utt, aux: True
		
		pipe.append(
			('pm_model', PhrasingMotifs('motifs', prompt_input_field, min_support=min_support,
					fit_filter=prompt_filter, verbosity=verbosity))
		)
		pipe.append(
			('pt_model', PromptTypes(prompt_field='motifs', ref_field='arcs_censored', 
									 prompt_transform_field='motifs__sink',
				 output_field=output_field, n_types=n_types,
				 svd__n_components=svd__n_components,
				 prompt_filter=prompt_filter, ref_filter=ref_filter,
				 prompt__tfidf_min_df=min_support,
				 prompt__tfidf_max_df=prompt__tfidf_max_df,
				 ref__tfidf_min_df=min_support,
				 ref__tfidf_max_df=ref__tfidf_max_df,
				 max_dist=max_dist,
				 random_state=random_state, verbosity=verbosity
			))
		)
		self.pipe = ConvokitPipeline(pipe)
		
	def fit(self, corpus, y=None):
		self.pipe.fit(corpus)
	
	def transform(self, corpus):
		return self.pipe.transform(corpus)
	
	def transform_utterance(self, utterance):
		if isinstance(utterance, str):
			utterance = Utterance(text=utterance)
			utterance.meta['is_question'] = True
		return self.pipe.transform_utterance(utterance)        
	
	def dump_models(self, model_dir, type_keys='default'):
		try:
			os.mkdir(model_dir)
		except:
			pass
		self.pipe.named_steps['pm_model'].dump_model(os.path.join(model_dir, 'pm_model'))
		self.pipe.named_steps['pt_model'].dump_model(os.path.join(model_dir, 'pt_model'), type_keys=type_keys)
	
	def load_models(self, model_dir, type_keys='default'):
		self.pipe.named_steps['pm_model'].load_model(os.path.join(model_dir, 'pm_model'))
		self.pipe.named_steps['pt_model'].load_model(os.path.join(model_dir, 'pt_model'), type_keys=type_keys)
	
	def print_top_phrasings(self, k):
		self.pipe.named_steps['pm_model'].print_top_phrasings(k)
	
	def display_type(self, type_id, corpus=None, type_key=None, k=10):
		self.pipe.named_steps['pt_model'].display_type(type_id, corpus=corpus, type_key=type_key, k=k)
	
	def refit_types(self, n_types, random_state=None, name=None):
		self.pipe.named_steps['pt_model'].refit_types(n_types, random_state, name)
	
	