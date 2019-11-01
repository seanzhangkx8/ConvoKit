from convokit import Corpus
from convokit.text_processing import TextProcessor, TextParser, TextToArcs
from convokit.phrasing_motifs import CensorNouns, QuestionSentences, PhrasingMotifs
from convokit.prompt_types import PromptTypes
from convokit.convokitPipeline import ConvokitPipeline
from convokit.transformer import Transformer
from convokit.model import Utterance

import os

class PromptTypeWrapper(Transformer):
	
	def __init__(self, output_field='prompt_types', n_types=8, use_prompt_motifs=True, root_only=True,
				questions_only=True, enforce_caps=True, recompute_all=False, min_support=100,
				 min_df=100, svd__n_components=25, max_df=.1,
					max_dist=.9,
				 random_state=None, verbosity=10000, 
				):
		self.use_motifs = use_prompt_motifs
		self.random_state=random_state
		pipe = [
			('parser', TextParser(verbosity=verbosity, 
				 input_filter=lambda utt, aux: recompute_all or (utt.get_info('parsed') is None))),
			('censor_nouns', CensorNouns('parsed_censored', 
				 input_filter=lambda utt, aux: recompute_all or (utt.get_info('parsed_censored') is None),
										 verbosity=verbosity)),
			('shallow_arcs', TextToArcs('arcs', input_field='parsed_censored',
				input_filter=lambda utt, aux: recompute_all or (utt.get_info('arcs') is None),
									   root_only=root_only, verbosity=verbosity))
			
		]
		
		if questions_only:
			pipe.append(
				('question_sentence_filter', QuestionSentences('question_arcs',
									input_field='arcs', 
								   input_filter=lambda utt, aux: recompute_all or utt.meta['is_question'],
									use_caps=enforce_caps, verbosity=verbosity))
			)
		
			prompt_input_field = 'question_arcs'
			prompt_filter = lambda utt, aux: utt.meta['is_question']
			ref_filter = lambda utt, aux: (not utt.meta['is_question']) and (utt.reply_to is not None)
		else:
			prompt_input_field = 'arcs'
			prompt_filter = lambda utt, aux: True
			ref_filter = lambda utt, aux: True
		if use_prompt_motifs:
			pipe.append(
				('pm_model', PhrasingMotifs('motifs', prompt_input_field, min_support=min_support,
						fit_filter=prompt_filter, verbosity=verbosity))
			)
			prompt_field = 'motifs'
			prompt_transform_field = 'motifs__sink'
		else:
			prompt_field = 'arcs'
			prompt_transform_field = 'arcs'
		pipe.append(
			('pt_model', PromptTypes(prompt_field=prompt_field, ref_field='arcs', 
									 prompt_transform_field=prompt_transform_field,
				 output_field=output_field, n_types=n_types,
				 svd__n_components=svd__n_components,
				 prompt_filter=prompt_filter, ref_filter=ref_filter,
				 prompt__tfidf_min_df=min_df,
				 prompt__tfidf_max_df=max_df,
				 ref__tfidf_min_df=min_df,
				 ref__tfidf_max_df=max_df,
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
		if self.use_motifs:
			self.pipe.named_steps['pm_model'].dump_model(os.path.join(model_dir, 'pm_model'))
		self.pipe.named_steps['pt_model'].dump_model(os.path.join(model_dir, 'pt_model'), type_keys=type_keys)
	
	def load_models(self, model_dir, type_keys='default'):
		if self.use_motifs:
			self.pipe.named_steps['pm_model'].load_model(os.path.join(model_dir, 'pm_model'))
		self.pipe.named_steps['pt_model'].load_model(os.path.join(model_dir, 'pt_model'), type_keys=type_keys)
	
	def print_top_phrasings(self, k):
		if self.use_motifs:
			self.pipe.named_steps['pm_model'].print_top_phrasings(k)
		else:
			print('phrasing motifs unavailable')
	
	def display_type(self, type_id, corpus=None, type_key=None, k=10):
		self.pipe.named_steps['pt_model'].display_type(type_id, corpus=corpus, type_key=type_key, k=k)
	
	def refit_types(self, n_types, random_state=None, name=None):
		self.pipe.named_steps['pt_model'].refit_types(n_types, random_state, name)
	
	