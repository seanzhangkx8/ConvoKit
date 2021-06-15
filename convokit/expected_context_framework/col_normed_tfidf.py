from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Normalizer, normalize
from scipy import sparse
import numpy as np 
import joblib
import os
import json

from convokit.transformer import Transformer

class ColNormedTfidfWrapper(Transformer):
    
    def __init__(self, input_field, output_field='col_normed_tfidf', **kwargs):
        self.tfidf_obj = ColNormedTfidf(**kwargs)
        self.input_field = input_field
        self.output_field = output_field
        if self.input_field == 'text':
            self.text_func = lambda x: x.text
        else:
            self.text_func = lambda x: x.meta[self.input_field]
    
    def fit(self, corpus, y=None, selector=lambda x: True):
        docs = [self.text_func(ut) for ut in corpus.iter_utterances(selector=selector)]
        self.tfidf_obj.fit(docs)
        return self
    
    def transform(self, corpus, selector=lambda x: True): 
        ids = []
        docs = []
        for ut in corpus.iter_utterances(selector=selector):
            ids.append(ut.id)
            docs.append(self.text_func(ut))
            ut.add_vector(self.output_field)
        vects = self.tfidf_obj.transform(docs)
        column_names = self.tfidf_obj.get_feature_names()
        corpus.set_vector_matrix(self.output_field, matrix=vects, ids=ids, columns=column_names)
        n_feats = np.array((vects>0).sum(axis=1)).flatten()
        for id, n in zip(ids, n_feats):
            corpus.get_utterance(id).meta[self.output_field + '__n_feats'] = n
        return corpus
    
    def fit_transform(self, corpus, y=None, selector=lambda x: True):
        self.fit(corpus, y, selector)
        return self.transform(corpus, selector)
    
    def get_vocabulary(self):
        return self.tfidf_obj.get_feature_names()
    
    def load_model(self, dirname):
        self.tfidf_obj.load(dirname)
    
    def dump_model(self, dirname):
        self.tfidf_obj.dump(dirname)

class ColNormedTfidf(TransformerMixin):
    
    def __init__(self, **kwargs):
        if 'token_pattern' in kwargs:
            self.tfidf_model = TfidfVectorizer(**kwargs)
        else:
            self.tfidf_model = TfidfVectorizer(token_pattern=r'(?u)(\S+)',**kwargs)
    
    def fit(self, X, y=None):
        tfidf_vects_raw = self.tfidf_model.fit_transform(X)
        self.col_norms = sparse.linalg.norm(tfidf_vects_raw, axis=0)
    
    def transform(self, X):
        tfidf_vects_raw = self.tfidf_model.transform(X)
        tfidf_vect = tfidf_vects_raw / self.col_norms 
        return tfidf_vect
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self):
        return self.tfidf_model.get_feature_names()
    
    def get_params(self, deep=True):
        return self.tfidf_model.get_params(deep=deep)
    
    def set_params(self, **params):
        return self.tfidf_model.set_params(**params)
    
    def load(self, dirname):
        self.tfidf_model = joblib.load(os.path.join(dirname, 'tfidf_model.joblib'))
        self.col_norms = np.load(os.path.join(dirname, 'col_norms.npy'))
    
    def dump(self, dirname):
        try:
            os.mkdir(dirname)
        except: pass
        np.save(os.path.join(dirname, 'col_norms.npy'), self.col_norms)
        joblib.dump(self.tfidf_model, os.path.join(dirname, 'tfidf_model.joblib'))  