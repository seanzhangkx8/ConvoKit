import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from IPython.display import display

from convokit.transformer import Transformer

class PromptTypes(Transformer):
    
    def __init__(self, prompt_field, ref_field, output_field, n_types=8,
                prompt_transform_field=None, ref_transform_field=None,
                prompt_filter=lambda utt, aux: True, ref_filter=lambda utt, aux: True,
                prompt_transform_filter=None, ref_transform_filter=None,
                prompt__tfidf_min_df=100, prompt__tfidf_max_df=.1,
                ref__tfidf_min_df=100, ref__tfidf_max_df=.1,
                snip_first_dim=True,
                svd__n_components=25, max_dist=.9,
                random_state=None, verbosity=0):
        
        self.prompt_embedding_model = {}
        self.type_models = {}
        self.train_results = {}
        self.train_types = {}
        
        self.prompt_field = prompt_field
        self.ref_field = ref_field
        self.prompt_filter = prompt_filter
        self.ref_filter = ref_filter
        
        if prompt_transform_field is None:
            self.prompt_transform_field = self.prompt_field
        else:
            self.prompt_transform_field = prompt_transform_field
        if prompt_transform_filter is None:
            self.prompt_transform_filter = self.prompt_filter
        else:
            self.prompt_transform_filter = prompt_transform_filter
        
        if ref_transform_field is None:
            self.ref_transform_field = self.ref_field
        else:
            self.ref_transform_field = ref_transform_field
        if ref_transform_filter is None:
            self.ref_transform_filter = self.ref_filter
        else:
            self.ref_transform_filter = ref_transform_filter
            
        self.output_field = output_field
        
        self.prompt__tfidf_min_df = prompt__tfidf_min_df
        self.prompt__tfidf_max_df = prompt__tfidf_max_df
        self.ref__tfidf_min_df = ref__tfidf_min_df
        self.ref__tfidf_max_df = ref__tfidf_max_df
        self.snip_first_dim = snip_first_dim
        self.svd__n_components = svd__n_components
        self.default_n_types = n_types
        self.random_state = random_state
        self.max_dist = max_dist
        self.verbosity = verbosity
    
    def fit(self, corpus, y=None):
        _, prompt_input, _, ref_input = self._get_pair_input(corpus, self.prompt_field, self.ref_field,
                                    self.prompt_filter, self.ref_filter)
        self.prompt_embedding_model = fit_prompt_embedding_model(prompt_input, ref_input,
                                self.snip_first_dim, self.prompt__tfidf_min_df, self.prompt__tfidf_max_df,
                                self.ref__tfidf_min_df, self.ref__tfidf_max_df,
                                self.svd__n_components, self.random_state, self.verbosity)
        self.train_results['prompt_ids'], self.train_results['prompt_vects'],\
            self.train_results['ref_ids'], self.train_results['ref_vects'] = self.get_embeddings(corpus) 
        self.refit_types(self.default_n_types, self.random_state)

        
    def transform(self, corpus):
        prompt_ids, prompt_vects, ref_ids, ref_vects = self.get_embeddings(corpus)
        corpus.set_vect_reprs(self.output_field + '__prompt_repr', prompt_ids, prompt_vects)
        corpus.set_vect_reprs(self.output_field + '__ref_repr', ref_ids, ref_vects)
        
        prompt_df, ref_df = self.get_type_assignments(prompt_ids, prompt_vects, ref_ids, ref_vects)
        prompt_dists, prompt_assigns = prompt_df[prompt_df.columns[:-1]].values, prompt_df['type_id'].values
        prompt_min_dists = prompt_dists.min(axis=1)
        ref_dists, ref_assigns = ref_df[ref_df.columns[:-1]].values, ref_df['type_id'].values
        ref_min_dists = ref_dists.min(axis=1)
        corpus.set_vect_reprs(self.output_field + '__prompt_dists.%s' % self.default_n_types, 
                                prompt_df.index, prompt_dists)
        corpus.set_vect_reprs(self.output_field + '__ref_dists.%s' % self.default_n_types, 
                                ref_df.index, ref_dists)
        for id, assign, dist in zip(prompt_df.index, prompt_assigns, prompt_min_dists):
            corpus.get_utterance(id).set_info(self.output_field + '__prompt_type.%s' % self.default_n_types, assign)
            corpus.get_utterance(id).set_info(self.output_field + '__prompt_type_dist.%s' % self.default_n_types, float(dist))
        for id, assign, dist in zip(ref_df.index, ref_assigns, ref_min_dists):
            corpus.get_utterance(id).set_info(self.output_field + '__ref_type.%s' % self.default_n_types, assign)
            corpus.get_utterance(id).set_info(self.output_field + '__ref_type_dist.%s' % self.default_n_types, float(dist))
        return corpus

    def transform_utterance(self, utterance):
    	if self.prompt_transform_filter(utterance, {}):
    		utterance = self._transform_utterance_side(utterance, 'prompt')
    	if self.ref_transform_filter(utterance, {}):
    		utterance = self._transform_utterance_side(utterance, 'ref')
    	return utterance
    	# utt_id = utterance.id 
    	# utt_input = utterance.get_info(self.prompt_transform_field)
    	# if isinstance(utt_input, list):
    	# 	utt_input = '\n'.join(utt_input)
    	# utt_ids, utt_vects = transform_embeddings(self.prompt_embedding_model, [utt_id], [utt_input], side='prompt')
    	# prompt_df = assign_prompt_types(self.type_models[self.default_n_types], utt_ids, utt_vects, self.max_dist)
    	# vals = prompt_df.values[0]
    	# dists = vals[:-1]
    	# min_dist = min(dists)
    	# assign = vals[-1]
    	# utterance.set_info(self.output_field + '__prompt_type.%s' % self.default_n_types, assign)
    	# utterance.set_info(self.output_field + '__prompt_type_dist.%s' % self.default_n_types, float(min_dist))
    	# utterance.set_info(self.output_field + '__prompt_dists.%s' % self.default_n_types, [float(x) for x in dists])
    	# utterance.set_info(self.output_field + '__prompt_repr', [float(x) for x in utt_vects[0]])
    	# return utterance

    def _transform_utterance_side(self, utterance, side):
    	if side == 'prompt':
    		input_field = self.prompt_transform_field
    	elif side == 'ref':
    		input_field = self.ref_transform_field
    	utt_id = utterance.id
    	utt_input = utterance.get_info(input_field)
    	if isinstance(utt_input, list):
    		utt_input = '\n'.join(utt_input)
    	utt_ids, utt_vects = transform_embeddings(self.prompt_embedding_model, [utt_id], [utt_input], side=side)
    	assign_df = assign_prompt_types(self.type_models[self.default_n_types], utt_ids, utt_vects, self.max_dist)
    	vals = assign_df.values[0]
    	dists = vals[:-1]
    	min_dist = min(dists)
    	assign = vals[-1]
    	utterance.set_info(self.output_field + '__%s_type.%s' % (side, self.default_n_types), assign)
    	utterance.set_info(self.output_field + '__%s_type_dist.%s' % (side, self.default_n_types), float(min_dist))
    	utterance.set_info(self.output_field + '__%s_dists.%s' % (side, self.default_n_types), [float(x) for x in dists])
    	utterance.set_info(self.output_field + '__%s_repr' % side, [float(x) for x in utt_vects[0]])
    	return utterance
        
    def refit_types(self, n_types, random_state=None, name=None):
        if name is None:
            key = n_types
        else:
            key = name
        if random_state is None:
            random_state = self.random_state
        self.type_models[key] = fit_prompt_type_model(self.prompt_embedding_model, n_types, random_state, self.max_dist, self.verbosity)
        prompt_df, ref_df = self.get_type_assignments(type_key=key)
        self.train_types[key] = {'prompt_df': prompt_df, 'ref_df': ref_df}

        
    def get_embeddings(self, corpus):
        prompt_ids, prompt_inputs = self._get_input(corpus, self.prompt_transform_field, 
                                                    self.prompt_transform_filter)
        ref_ids, ref_inputs = self._get_input(corpus, self.ref_transform_field, self.ref_transform_filter)
        prompt_ids, prompt_vects = transform_embeddings(self.prompt_embedding_model, 
                                                        prompt_ids, prompt_inputs, 
                                                        side='prompt')
        ref_ids, ref_vects = transform_embeddings(self.prompt_embedding_model, 
                                                        ref_ids, ref_inputs, 
                                                        side='ref')
        return prompt_ids, prompt_vects, ref_ids, ref_vects

    
    def get_type_assignments(self, prompt_ids=None, prompt_vects=None, 
                             ref_ids=None, ref_vects=None, type_key=None):
        if prompt_ids is None:
            prompt_ids, prompt_vects, ref_ids, ref_vects = [self.train_results[k] for k in
                                        ['prompt_ids', 'prompt_vects', 'ref_ids', 'ref_vects']]
        if type_key is None:
            type_key = self.default_n_types
        prompt_df = assign_prompt_types(self.type_models[type_key], prompt_ids, prompt_vects, self.max_dist)
        ref_df = assign_prompt_types(self.type_models[type_key], ref_ids, ref_vects, self.max_dist)
        return prompt_df, ref_df
    
        
    def display_type(self, type_id, corpus=None, type_key=None, k=10):
        if type_key is None:
            type_key = self.default_n_types
        prompt_df = self.type_models[type_key]['prompt_df']
        ref_df = self.type_models[type_key]['ref_df']
        
        top_prompt = prompt_df[prompt_df.type_id == type_id].sort_values(type_id).head(k)
        top_ref = ref_df[ref_df.type_id == type_id].sort_values(type_id).head(k)
        print('top prompt:')
        display(top_prompt)
        print('top response:')
        display(top_ref)
        
        if corpus is not None:
            prompt_df = self.train_types[type_key]['prompt_df']
            ref_df = self.train_types[type_key]['ref_df']
            top_prompt = prompt_df[prompt_df.type_id == type_id].sort_values(type_id).head(k).index
            top_ref = ref_df[ref_df.type_id == type_id].sort_values(type_id).head(k).index
            print('top prompts:')
            for utt in top_prompt:
                print(utt, corpus.get_utterance(utt).text)
                print(corpus.get_utterance(utt).get_info(self.prompt_transform_field))
                print()
            print('top responses:')
            for utt in top_ref:
                print(utt, corpus.get_utterance(utt).text)
                print(corpus.get_utterance(utt).get_info(self.ref_transform_field))
                print()
    
    def dump_model(self, model_dir, type_keys='default', dump_train_corpus=True):
        if self.verbosity > 0:
            print('dumping embedding model')
        if not os.path.exists(model_dir):
            try:
                os.mkdir(model_dir)
            except:
                pass
        for k in ['prompt_tfidf_model', 'ref_tfidf_model', 'svd_model']:
            joblib.dump(self.prompt_embedding_model[k], 
                       os.path.join(model_dir, k + '.joblib'))
       
        for k in ['U_prompt', 'U_ref']:
            np.save(os.path.join(model_dir, k), self.prompt_embedding_model[k])
        
        if dump_train_corpus:
            if self.verbosity > 0:
                print('dumping training embeddings')
            for k in ['prompt_ids', 'prompt_vects', 'ref_ids', 'ref_vects']:
                np.save(os.path.join(model_dir, 'train_' + k), self.train_results[k])
        
        if type_keys == 'default':
            to_dump = [self.default_n_types]
        elif type_keys == 'all':
            to_dump = self.type_models.keys()
        else:
            to_dump = type_keys
        for key in to_dump:
            if self.verbosity > 0:
                print('dumping type model', key)
            type_model = self.type_models[key]
            joblib.dump(type_model['km_model'], os.path.join(model_dir, 'km_model.%s.joblib' % key))
            for k in ['prompt_df', 'ref_df']:
                type_model[k].to_csv(os.path.join(model_dir, '%s.%s.tsv' % (k, key)), sep='\t')
            if dump_train_corpus:
                train_types = self.train_types[key]
                for k in ['prompt_df', 'ref_df']:
                    train_types[k].to_csv(os.path.join(model_dir, 'train_%s.%s.tsv' % (k, key)), sep='\t')
    
    def load_model(self, model_dir, type_keys='default', load_train_corpus=True):
        if self.verbosity > 0:
            print('loading embedding model')
        for k in ['prompt_tfidf_model', 'ref_tfidf_model', 'svd_model']:
            self.prompt_embedding_model[k] = joblib.load(os.path.join(model_dir, k + '.joblib'))
        for k in ['U_prompt', 'U_ref']:
            self.prompt_embedding_model[k] = np.load(os.path.join(model_dir, k + '.npy'))
        
        if load_train_corpus:
            if self.verbosity > 0:
                print('loading training embeddings')
            for k in ['prompt_ids', 'prompt_vects', 'ref_ids', 'ref_vects']:
                self.train_results[k] = np.load(os.path.join(model_dir, 'train_' + k + '.npy'))
        
        if type_keys == 'default':
            to_load = [self.default_n_types]
        elif type_keys == 'all':
            to_load = [x.replace('km_model.','').replace('.joblib','')
                      for x in os.listdir(model_dir) if x.startswith('km_model')]
        else:
            to_load = type_keys
        for key in to_load:
            try:
                key = int(key)
            except: pass
            if self.verbosity > 0:
                print('loading type model', key)
            self.type_models[key] = {} # this should be an int-ish
            self.type_models[key]['km_model'] = joblib.load(
                os.path.join(model_dir, 'km_model.%s.joblib' % key))
            
            for k in ['prompt_df', 'ref_df']:
                self.type_models[key][k] =\
                    pd.read_csv(os.path.join(model_dir, '%s.%s.tsv' % (k, key)), sep='\t', index_col=0)
                self.type_models[key][k].columns = [int(x) for x in self.type_models[key][k].columns[:-1]]\
                    + ['type_id']
            if load_train_corpus:
                self.train_types[key] = {}
                for k in ['prompt_df', 'ref_df']:
                    self.train_types[key][k] = pd.read_csv(
                        os.path.join(model_dir, 'train_%s.%s.tsv' % (k, key)), sep='\t', index_col=0
                    )
                    self.train_types[key][k].columns = \
                        [int(x) for x in self.train_types[key][k].columns[:-1]] + ['type_id']

    def _get_input(self, corpus, field, filter_fn, check_nonempty=True, aux_input={}):
        ids = []
        inputs = []
        for utterance in corpus.iter_utterances():
            input = utterance.get_info(field)
            if isinstance(input, list):
                input = '\n'.join(input)
            if filter_fn(utterance, aux_input)\
                and ((not check_nonempty) or (len(input) > 0)):
                ids.append(utterance.id)
                inputs.append(input)
        return ids, inputs

    # def _get_input(self, corpus, field, filter_fn, check_nonempty=True, aux_input={}):
    #     ids = []
    #     inputs = []
    #     for id in corpus.get_utterance_ids():
    #         input = corpus.get_info(id, field)
    #         if isinstance(input, list):
    #             input = '\n'.join(input)
    #         if filter_fn(id, corpus, aux_input)\
    #             and ((not check_nonempty) or (len(input) > 0)):
    #             ids.append(id)
    #             inputs.append(input)
    #     return ids, inputs
    def _get_pair_input(self, corpus, prompt_field, ref_field, 
              prompt_filter=lambda x,y,z: True, ref_filter=lambda x,y,z: True, 
              check_nonempty=True, aux_input={}):
        prompt_ids = []
        prompt_utts = []
        ref_ids = []
        ref_utts = []
        for ref_utt in corpus.iter_utterances():
            if ref_utt.reply_to is None:
                continue
            prompt_utt_id = ref_utt.reply_to
            try:
            	prompt_utt = corpus.get_utterance(prompt_utt_id)
            except: 
            	continue
            if prompt_filter(prompt_utt, aux_input) \
                and ref_filter(ref_utt, aux_input):

                prompt_input = prompt_utt.get_info(prompt_field)
                ref_input = ref_utt.get_info(ref_field)
                
                if (prompt_input is None) or (ref_input is None):
                	continue

                if isinstance(prompt_input, list):
                     prompt_input = '\n'.join(prompt_input)
                if isinstance(ref_input, list):
                     ref_input = '\n'.join(ref_input)

                if (not check_nonempty) or ((len(prompt_input) > 0) and (len(ref_input) > 0)):
                    prompt_ids.append(prompt_utt.id)
                    prompt_utts.append(prompt_input)
                    ref_ids.append(ref_utt.id)
                    ref_utts.append(ref_input)
        return prompt_ids, prompt_utts, ref_ids, ref_utts        
    # def _get_pair_input(self, corpus, prompt_field, ref_field, 
    #           prompt_filter=lambda x,y,z: True, ref_filter=lambda x,y,z: True, 
    #           check_nonempty=True, aux_input={}):
    #     prompt_ids = []
    #     prompt_utts = []
    #     ref_ids = []
    #     ref_utts = []
    #     for ref_utt_id in corpus.get_utterance_ids():
    #         ref_utt = corpus.get_utterance(ref_utt_id)
    #         if ref_utt.reply_to is None:
    #             continue
    #         prompt_utt_id = ref_utt.reply_to
    #         if prompt_filter(prompt_utt_id, corpus, aux_input) \
    #             and ref_filter(ref_utt_id, corpus, aux_input):
    #             prompt_input = corpus.get_info(prompt_utt_id, prompt_field)
    #             ref_input = corpus.get_info(ref_utt_id, ref_field)
                
    #             if isinstance(prompt_input, list):
    #                  prompt_input = '\n'.join(prompt_input)
    #             if isinstance(ref_input, list):
    #                  ref_input = '\n'.join(ref_input)

    #             if (not check_nonempty) or ((len(prompt_input) > 0) and (len(ref_input) > 0)):
    #                 prompt_ids.append(prompt_utt_id)
    #                 prompt_utts.append(prompt_input)
    #                 ref_ids.append(ref_utt_id)
    #                 ref_utts.append(ref_input)
    #     return prompt_ids, prompt_utts, ref_ids, ref_utts
# standalone functions


def fit_prompt_embedding_model(prompt_input, ref_input, snip_first_dim=True,
            prompt__tfidf_min_df=100, prompt__tfidf_max_df=.1,
            ref__tfidf_min_df=100, ref__tfidf_max_df=.1,
            svd__n_components=25, random_state=None, verbosity=0):
    
    if verbosity > 0:
        print('fitting %d input pairs' % len(prompt_input))
        print('fitting ref tfidf model')
    ref_tfidf_model = TfidfVectorizer(
        min_df=ref__tfidf_min_df,
        max_df=ref__tfidf_max_df,
        binary=True,
        token_pattern=r'(?u)(\S+)'
    )
    ref_vect = ref_tfidf_model.fit_transform(ref_input)
    
    if verbosity > 0:
        print('fitting prompt tfidf model')
    prompt_tfidf_model = TfidfVectorizer(
        min_df=prompt__tfidf_min_df,
        max_df=prompt__tfidf_max_df,
        binary=True,
        token_pattern=r'(?u)(\S+)'
    )
    prompt_vect = prompt_tfidf_model.fit_transform(prompt_input)
    
    if verbosity > 0:
        print('fitting svd model')
    svd_model = TruncatedSVD(n_components=svd__n_components, random_state=random_state, algorithm='arpack')
   
    U_ref = svd_model.fit_transform(normalize(ref_vect.T))
    s = svd_model.singular_values_
    U_ref /= s
    U_prompt = (svd_model.components_ * normalize(prompt_vect, axis=0) / s[:, np.newaxis]).T 
    
    if snip_first_dim:
        U_prompt = U_prompt[:, 1:]
        U_ref = U_ref[:, 1:]
    U_prompt_norm = normalize(U_prompt)
    U_ref_norm = normalize(U_ref)
    
    return {'prompt_tfidf_model': prompt_tfidf_model, 'ref_tfidf_model': ref_tfidf_model,
           'svd_model': svd_model, 'U_prompt': U_prompt_norm, 'U_ref': U_ref_norm}

def transform_embeddings(model, ids, input, side='prompt', filter_empty=True):
    tfidf_vects = normalize(model['%s_tfidf_model' % side].transform(input), norm='l1')
    mask = np.array(tfidf_vects.sum(axis=1)).flatten() > 0
    vects = normalize(tfidf_vects * model['U_%s' % side])
    if filter_empty:
        ids = np.array(ids)[mask]
        vects = vects[mask]
    return ids, vects

def fit_prompt_type_model(model, n_types, random_state=None, max_dist=0.9, verbosity=0):
    if verbosity > 0:
        print('fitting %d prompt types' % n_types)
    km = KMeans(n_clusters=n_types, random_state=random_state)
    km.fit(model['U_prompt'])
    prompt_dists = km.transform(model['U_prompt'])
    prompt_clusters = km.predict(model['U_prompt'])
    prompt_clusters[prompt_dists.min(axis=1) >= max_dist] = -1
    ref_dists = km.transform(model['U_ref'])
    ref_clusters = km.predict(model['U_ref'])
    ref_clusters[ref_dists.min(axis=1) >= max_dist] = -1
    
    prompt_df = pd.DataFrame(index=model['prompt_tfidf_model'].get_feature_names(),
                          data=np.hstack([prompt_dists, prompt_clusters[:,np.newaxis]]),
                          columns=list(range(n_types)) + ['type_id'])
    ref_df = pd.DataFrame(index=model['ref_tfidf_model'].get_feature_names(),
                          data=np.hstack([ref_dists, ref_clusters[:,np.newaxis]]),
                          columns=list(range(n_types)) + ['type_id'])
    return {'km_model': km, 
           'prompt_df': prompt_df, 'ref_df': ref_df}

def assign_prompt_types(model, ids, vects, max_dist=0.9):
    dists = model['km_model'].transform(vects)
    clusters = model['km_model'].predict(vects)
    dist_mask = dists.min(axis=1) >= max_dist
    clusters[ dist_mask] = -1
    df = pd.DataFrame(index=ids, data=np.hstack([dists,clusters[:,np.newaxis]]),
                     columns=list(range(dists.shape[1])) + ['type_id'])
    return df
