import os
import pandas as pd
import numpy as np


from sklearn.preprocessing import Normalizer, normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances, paired_distances
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from scipy import sparse
import joblib
import json

from convokit.transformer import Transformer

class ExpectedContextModelWrapper(Transformer):
    
    def __init__(self, context_field,output_prefix,
                 vect_field, context_vect_field=None,
                n_svd_dims=25, snip_first_dim=True, n_clusters=8, cluster_on='utts',
                model=None, random_state=None, cluster_random_state=None):
        if model is not None:
            in_model = model.ec_model
        else:
            in_model = None
        self.ec_model = ExpectedContextModel(model=in_model,
            n_svd_dims=n_svd_dims, snip_first_dim=snip_first_dim, n_clusters=n_clusters, cluster_on=cluster_on,
            random_state=random_state, cluster_random_state=cluster_random_state)
        self.context_field = context_field
        if context_field == 'reply_to':
            self.context_func = lambda x: x.reply_to
        else:
            self.context_func = lambda x: x.meta.get(context_field, None)
        self.output_prefix = output_prefix
        self.vect_field = vect_field
        self.context_vect_field = context_vect_field
        if self.context_vect_field is None:
            self.context_vect_field = vect_field
    
    def fit(self, corpus, y=None, selector=lambda x: True, context_selector=lambda x: True):
    
        id_to_idx = corpus.get_vector_matrix(self.vect_field).ids_to_idx
        context_id_to_idx = corpus.get_vector_matrix(self.context_vect_field).ids_to_idx
        
        
        ids = []
        context_ids = []
        mapping_ids = []
        context_mapping_ids = []
        for ut in corpus.iter_utterances(selector=selector):
            ids.append(ut.id)
            context_id = self.context_func(ut)
            if context_id is not None:
                try:
                    if context_selector(corpus.get_utterance(context_id)):
                        try:
                            mapping_ids.append(ut.id)
                            context_mapping_ids.append(context_id)
                        except: continue
                except:
                    continue
                    
        for ut in corpus.iter_utterances(selector=context_selector):
            context_ids.append(ut.id)

        id_to_idx = {id: i for i, id in enumerate(ids)}
        context_id_to_idx = {id: i for i, id in enumerate(context_ids)}
        mapping_idxes = [id_to_idx[x] for x in mapping_ids]
        context_mapping_idxes = [context_id_to_idx[x] for x in context_mapping_ids]
        
        utt_vects = corpus.get_vectors(self.vect_field, ids)
        context_utt_vects = corpus.get_vectors(self.context_vect_field, context_ids)
        mapping_table = np.vstack([mapping_idxes, context_mapping_idxes]).T
        self.mapping_table = mapping_table
        terms = corpus.get_vector_matrix(self.vect_field).columns
        context_terms = corpus.get_vector_matrix(self.context_vect_field).columns
        self.ec_model.fit(utt_vects, context_utt_vects, mapping_table,
                         terms, context_terms, utt_ids=ids, context_utt_ids=context_ids)
            
    def _get_matrix(self, corpus, field, selector):
        ids = [ut.id for ut in corpus.iter_utterances(selector=selector)
              if field in ut.vectors]
        utt_vects = corpus.get_vectors(field, ids)
        return ids, utt_vects
    
    def _add_vector(self, corpus, field, ids):
        for id in ids:
            corpus.get_utterance(id).add_vector(field)
    
    def transform(self, corpus, selector=lambda x: True):

        ids, utt_vects = self._get_matrix(corpus, self.vect_field, selector)
        utt_reprs = self.ec_model.transform(utt_vects)
        corpus.set_vector_matrix(self.output_prefix + '_repr', matrix=utt_reprs,
                                ids=ids)
        self._add_vector(corpus, self.output_prefix + '_repr', ids)
        self.compute_utt_ranges(corpus, selector)
        self.compute_clusters(corpus, selector)
        return corpus
    
    def compute_utt_ranges(self, corpus, selector=lambda x: True):
        
        ids, utt_vects = self._get_matrix(corpus, self.vect_field, selector)
        ranges = self.ec_model.compute_utt_ranges(utt_vects)
        for id, r in zip(ids, ranges):
            corpus.get_utterance(id).meta[self.output_prefix + '_range'] = r
        return ranges
    
    def transform_context_utts(self, corpus, selector=lambda x: True):
        ids, context_utt_vects = self._get_matrix(corpus, self.context_vect_field, selector)
        context_utt_reprs = self.ec_model.transform_context_utts(context_utt_vects)
        corpus.set_vector_matrix(self.output_prefix + '_context_repr', matrix=context_utt_reprs,
                                ids=ids)
        self._add_vector(corpus, self.output_prefix + '_context_repr', ids)
        self.compute_clusters(corpus, selector, is_context=True)
        return corpus
    
    def fit_clusters(self, n_clusters, random_state='default'):
        if random_state == 'default':
            random_state = self.ec_model.random_state
        self.ec_model.fit_clusters(n_clusters, random_state)
    
    def compute_clusters(self, corpus, selector=lambda x: True, is_context=False, cluster_suffix=''):
        if is_context:
            ids, reprs = self._get_matrix(corpus, self.output_prefix + '_context_repr', selector)
        else:
            ids, reprs = self._get_matrix(corpus, self.output_prefix + '_repr', selector)
        cluster_df = self.ec_model.transform_clusters(reprs, ids)
        if is_context:
            cluster_field = self.output_prefix + '_context_clustering'
        else:
            cluster_field = self.output_prefix + '_clustering'
        cluster_field += cluster_suffix
        for id, entry in cluster_df.iterrows():
            for k, v in entry.to_dict().items():
                corpus.get_utterance(id).meta[cluster_field + '.' + k] = v
        return cluster_df
    
    def set_cluster_names(self, cluster_names):
        self.ec_model.set_cluster_names(cluster_names)
    
    def get_cluster_names(self):
        return self.ec_model.get_cluster_names()
    
    def print_clusters(self, k=10, max_chars=1000, corpus=None):
        n_clusters = self.ec_model.n_clusters
        cluster_obj = self.ec_model.clustering
        for i in range(n_clusters):
            print('CLUSTER', i, self.ec_model.get_cluster_names()[i])
            print('---')
            print('terms')
            term_subset = cluster_obj['terms'][cluster_obj['terms'].cluster_id_ == i].sort_values('cluster_dist').head(k)
            print(term_subset[['cluster_dist']])
            print()
            print('context terms')
            context_term_subset = cluster_obj['context_terms'][cluster_obj['context_terms'].cluster_id_ == i].sort_values('cluster_dist').head(k)
            print(context_term_subset[['cluster_dist']])
            print()
            if corpus is None: continue
            print()
            print('utterances')
            utt_subset = cluster_obj['utts'][cluster_obj['utts'].cluster_id_ == i].drop_duplicates('cluster_dist').sort_values('cluster_dist').head(k)
            for id, row in utt_subset.iterrows():
                print('>', id, '%.3f' % row.cluster_dist, corpus.get_utterance(id).text[:max_chars])
            print()
            print('context utterances')
            context_utt_subset = cluster_obj['context_utts'][cluster_obj['context_utts'].cluster_id_ == i].drop_duplicates('cluster_dist').sort_values('cluster_dist').head(k)
            for id, row in context_utt_subset.iterrows():
                print('>>', id, '%.3f' % row.cluster_dist, corpus.get_utterance(id).text[:max_chars])
            print('\n====\n')
    
    def print_cluster_stats(self):
        cluster_obj = self.ec_model.clustering
        return pd.concat([
            cluster_obj[k].cluster.value_counts(normalize=True).rename(k).sort_index()
            for k in ['utts', 'terms', 'context_utts', 'context_terms']
        ], axis=1)
    
    def get_terms(self):
        return self.ec_model.terms 

    def get_term_ranges(self):
        return self.ec_model.term_ranges

    def get_term_reprs(self):
        return self.ec_model.term_reprs

    def get_context_terms(self):
        return self.ec_model.context_terms

    def get_context_term_reprs(self):
        return self.ec_model.context_term_reprs

    def get_clustering(self):
        return self.ec_model.clustering

    def get_cluster_names(self):
        return self.ec_model.get_cluster_names()

    def load(self, dirname):
        self.ec_model.load(dirname)
    
    def dump(self, dirname, dump_clustering=True):
        self.ec_model.dump(dirname, dump_clustering)

# def snip(vects, dim=None, snip_first_dim=True):
#     if dim is None:
#         dim = vects.shape[1]
#     return normalize(vects[:,int(snip_first_dim):dim])

class ExpectedContextModel:
    
    def __init__(self, n_svd_dims=25, snip_first_dim=True, n_clusters=8,
                     context_U=None, context_V=None, context_s=None,
                     model=None,
                     context_terms=None, cluster_on='utts',
                     random_state=None, cluster_random_state=None):
        
        self.n_svd_dims = n_svd_dims
        self.snip_first_dim = snip_first_dim
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.cluster_random_state = cluster_random_state
        self.cluster_on = cluster_on
        
        if (context_U is None) and (model is None):
            self.fitted_context = False
        elif (model is not None):
            self.fitted_context = True
            self.n_svd_dims = model.n_svd_dims
            self.context_U = model.context_U
            self.train_context_reprs = self._snip(self.context_U, self.snip_first_dim)
            self.context_V = model.context_V
            self.context_term_reprs = self._snip(self.context_V, self.snip_first_dim)
            self.context_s = model.context_s
            self.context_terms = self._get_default_ids(model.context_terms, len(self.context_V))
        elif (context_U is not None):
            self.fitted_context = True
            self.n_svd_dims = context_U.shape[1]
            self.context_U = context_U
            self.train_context_reprs = self._snip(self.context_U, self.snip_first_dim)
            
            self.context_V = context_V
            self.context_term_reprs = self._snip(self.context_V, self.snip_first_dim)
            
            self.context_s = context_s
            self.context_terms = self._get_default_ids(context_terms, len(self.context_V))

        self.terms = None 
        self.clustering = {}
            
    def fit_context_utts(self, context_utt_vects, 
                    context_terms=None):
        self.context_U, self.context_s, self.context_V = \
            randomized_svd(context_utt_vects, n_components=self.n_svd_dims,
                          random_state=self.random_state)
        self.train_context_reprs = self._snip(self.context_U, self.snip_first_dim)
        
        self.context_V = self.context_V.T
        self.context_term_reprs = self._snip(self.context_V, self.snip_first_dim)
        
        self.context_terms = self._get_default_ids(context_terms, len(self.context_V))
        self.fitted_context = True
                                                
    def fit(self, utt_vects, context_utt_vects, utt_context_pairs,
            terms=None, context_terms=None,
            refit_context=False, fit_clusters=True, n_clusters=None, cluster_random_state=None,
           utt_ids=None, context_utt_ids=None):
        if (not self.fitted_context) or refit_context:
            self.fit_context_utts(context_utt_vects, context_terms)
        
        self.terms = self._get_default_ids(terms, utt_vects.shape[1])
        
        utt_vect_subset = utt_vects[utt_context_pairs[:,0]]
        context_repr_subset = self.context_U[utt_context_pairs[:,1]]
        self.term_reprs_full = utt_vect_subset.T * context_repr_subset / self.context_s
        self.term_reprs = self._snip(self.term_reprs_full, snip_first_dim=self.snip_first_dim)
        self.train_utt_reprs = self.transform(utt_vects)
        
        full_dists = cosine_distances(
                self.term_reprs,
                self._snip(context_repr_subset, snip_first_dim=self.snip_first_dim)
            )
        weights = normalize(np.array(utt_vect_subset > 0), norm='l1', axis=0)
        clipped_dists = np.clip(full_dists, None, 1)
        self.term_ranges = (clipped_dists * weights.T).sum(axis=1)
        if fit_clusters:
            if self.n_clusters is None:
                self.n_clusters = n_clusters
            if self.cluster_random_state is None:
                self.cluster_random_state = cluster_random_state
            self.fit_clusters(self.n_clusters, self.cluster_random_state,
                             utt_ids=utt_ids, context_utt_ids=context_utt_ids)
        
    def transform(self, utt_vects):
        return self._snip(utt_vects * self.term_reprs_full / self.context_s, self.snip_first_dim)
        
    def compute_utt_ranges(self, utt_vects):
        return np.dot(normalize(utt_vects, norm='l1'), self.term_ranges)
    
    def transform_context_utts(self, context_utt_vects):
        return self._snip(context_utt_vects * self.context_V / self.context_s, self.snip_first_dim)  
    
    def fit_clusters(self, n_clusters=8, random_state='default', utt_ids=None, context_utt_ids=None):
        if random_state == 'default':
            random_state = self.cluster_random_state
        km_obj = ClusterWrapper(n_clusters=n_clusters, random_state=random_state)
        if self.cluster_on == 'terms':
            km_obj.fit(self.term_reprs)
        elif self.cluster_on == 'utts':
            km_obj.fit(self.train_utt_reprs)
        self.clustering['km_obj'] = km_obj
        self.clustering['utts'] = km_obj.transform(self.train_utt_reprs, utt_ids)
        self.clustering['terms'] = km_obj.transform(self.term_reprs, self.terms)
        self.clustering['context_utts'] = km_obj.transform(self.train_context_reprs, context_utt_ids)
        self.clustering['context_terms'] = km_obj.transform(self.context_term_reprs, self.context_terms)
    
    def transform_clusters(self, reprs, ids=None):
        return self.clustering['km_obj'].transform(reprs, ids)
    
    def set_cluster_names(self, cluster_names):
        cluster_names = np.array(cluster_names)
        self.clustering['km_obj'].set_cluster_names(cluster_names)
        for k in ['utts','terms','context_utts','context_terms']:
            self.clustering[k]['cluster'] = cluster_names[self.clustering[k].cluster_id_]

    def get_cluster_names(self):
        return self.clustering['km_obj'].cluster_names
    
    def load(self, dirname):
        with open(os.path.join(dirname, 'meta.json')) as f:
            meta_dict = json.load(f)
        self.n_svd_dims = meta_dict['n_svd_dims']
        self.random_state = meta_dict['random_state']
        self.snip_first_dim = meta_dict['snip_first_dim']
        self.cluster_on = meta_dict['cluster_on']
        
        self.context_U = np.load(os.path.join(dirname, 'context_U.npy'))
        self.train_context_reprs = self._snip(self.context_U, self.snip_first_dim)
        self.context_V = np.load(os.path.join(dirname, 'context_V.npy'))
        self.context_term_reprs = self._snip(self.context_V, self.snip_first_dim)
        self.context_s = np.load(os.path.join(dirname, 'context_s.npy'))
        self.context_terms = np.load(os.path.join(dirname, 'context_terms.npy'))
        self.terms = np.load(os.path.join(dirname, 'terms.npy'))
        self.term_reprs_full = np.load(os.path.join(dirname, 'term_reprs.npy'))
        self.term_reprs = self._snip(self.term_reprs_full, self.snip_first_dim)
        self.term_ranges = np.load(os.path.join(dirname, 'term_ranges.npy'))
        self.train_utt_reprs = np.load(os.path.join(dirname, 'train_utt_reprs.npy'))
        
        try:
            km_obj = ClusterWrapper(self.n_clusters)
            km_obj.load(dirname)
            self.clustering['km_obj'] = km_obj
            for k in ['utts','terms','context_utts','context_terms']:
                self.clustering[k] = pd.read_csv(os.path.join(dirname, 'clustering_%s.tsv' % k),
                                                sep='\t', index_col=0)
        except Exception as e:
            pass
    
    def dump(self, dirname, dump_clustering=True):
        try:
            os.mkdir(dirname)
        except: 
            pass
        with open(os.path.join(dirname, 'meta.json'), 'w') as f:
            json.dump({'n_svd_dims': self.n_svd_dims, 
                      'random_state': self.random_state,
                      'snip_first_dim': self.snip_first_dim,
                      'cluster_on': self.cluster_on}, f)
        for name, obj in [('context_U', self.context_U),
                         ('context_V', self.context_V),
                         ('context_s', self.context_s),
                         ('context_terms', self.context_terms),
                         ('terms',  self.terms),
                         ('term_reprs', self.term_reprs_full),
                         ('term_ranges', self.term_ranges),
                         ('train_utt_reprs', self.train_utt_reprs)]:
            np.save(os.path.join(dirname, name + '.npy'), obj)
        if dump_clustering and (len(self.clustering) > 0):
            self.clustering['km_obj'].dump(dirname)
            for k in ['utts','terms','context_utts','context_terms']:
                self.clustering[k].to_csv(os.path.join(dirname, 'clustering_%s.tsv' % k), sep='\t')
    
    def _get_default_ids(self, ids, n):
        if ids is None:
            return np.arange(n)
        else: return ids

    def _snip(self, vects, snip_first_dim=True, dim=None):
        if dim is None:
            dim = vects.shape[1]
        return normalize(vects[:,int(snip_first_dim):dim])

class ClusterWrapper:
    
    def __init__(self, n_clusters, cluster_names=None, random_state=None):
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        self.cluster_names = np.arange(n_clusters)
        if cluster_names is not None:
            self.cluster_names = cluster_names
        self.km_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.km_df = None
    
    def fit(self, vects, ids=None):
        
        self.km_model.fit(vects)
        self.km_df = self.transform(vects, ids)
        
    def set_cluster_names(self, names):
        self.cluster_names = np.array(names)
        if self.km_df is not None:
            self.km_df['cluster'] = self.cluster_names[self.km_df.cluster_id_]
    
    def transform(self, vects, ids=None):
        if ids is None:
            ids = np.arange(len(vects))
        km_df = self._get_km_assignment_df(self.km_model,
                     vects, ids, self.cluster_names)
        return km_df
    
    def _get_km_assignment_df(self, km, vects, ids, cluster_names):
        dists = km.transform(vects)
        min_dist = dists[np.arange(len(dists)), dists.argmin(axis=1)]
        cluster_assigns = km.predict(vects)
        cluster_assign_names = cluster_names[cluster_assigns]
        df = pd.DataFrame({'index': ids,  
                          'cluster_id_': cluster_assigns,
                          'cluster': cluster_assign_names,
                          'cluster_dist': min_dist}).set_index('index')
        return df
    
    def load(self, dirname):
        with open(os.path.join(dirname, 'cluster_meta.json')) as f:
            meta_dict = json.load(f)
        self.n_clusters = meta_dict['n_clusters']
        self.random_state = meta_dict['random_state']
        
        self.km_df = pd.read_csv(os.path.join(dirname, 'cluster_km_df.tsv'),
                                sep='\t', index_col=0)
        self.cluster_names = np.load(os.path.join(dirname, 'cluster_names.npy'))
        self.km_model = joblib.load(os.path.join(dirname, 'km_model.joblib'))
    
    def dump(self, dirname):
        try:
            os.mkdir(dirname)
        except: pass
        with open(os.path.join(dirname, 'cluster_meta.json'), 'w') as f:
            json.dump({'n_clusters': self.n_clusters,
                      'random_state': self.random_state}, f)
        self.km_df.to_csv(os.path.join(dirname, 'cluster_km_df.tsv'), sep='\t')
        np.save(os.path.join(dirname, 'cluster_names.npy'), self.cluster_names)
        joblib.dump(self.km_model, os.path.join(dirname, 'km_model.joblib'))