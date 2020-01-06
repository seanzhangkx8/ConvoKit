from convokit import Corpus, CorpusObject, Transformer
from typing import Callable, List
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

class BoWClassifier(Transformer):
    def __init__(self, obj_type: str, vectorizer=None, vector_name="bow_vector",
                 text_func: Callable[[CorpusObject], str] = lambda utt: utt.text,
                 labeller: Callable[[CorpusObject], bool] = lambda x: True,
                 selector: Callable[[CorpusObject], bool] = lambda x: True,
                 clf=None, clf_feat_name: str = "prediction", clf_prob_feat_name: str = "pred_score"):

        if vectorizer is None:
            print("Initializing default unigram CountVectorizer...")
            self.vectorizer = CV(decode_error='ignore', min_df=10, max_df=.5,
                                 ngram_range=(1, 1), binary=False, max_features=15000)
        else:
            self.vectorizer = vectorizer

        if clf is None:
            print("Initializing default classification model (standard scaled logistic regression)")
            clf = Pipeline([("standardScaler", StandardScaler(with_mean=False)),
                            ("logreg", LogisticRegression(solver='liblinear'))])

        self.obj_type = obj_type
        self.labeller = labeller
        self.selector = selector
        self.clf = clf
        self.clf_feat_name = clf_feat_name
        self.clf_prob_feat_name = clf_prob_feat_name
        self.vector_name = vector_name
        self.text_func = text_func

    def fit(self, corpus: Corpus, y=None):
        # collect texts for vectorization
        docs = []
        y = []
        for obj in corpus.iter_objs(self.obj_type, self.selector):
            docs.append(self.text_func(obj))
            y.append(self.labeller(obj))

        X = self.vectorizer.fit_transform(docs)
        self.clf.fit(X, y)

    def transform(self, corpus: Corpus) -> Corpus:
        objs = []
        docs = []
        for obj in corpus.iter_objs(self.obj_type):
            if self.selector(obj):
                objs.append(obj)
                obj.meta[self.vector_name] = self.vectorizer.transform([self.text_func(obj)])
                docs.append(self.text_func(obj))
            else:
                obj.meta[self.vector_name] = None

        X = self.vectorizer.transform(docs)
        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            obj = objs[idx]
            obj.add_meta(self.clf_feat_name, clf)
            obj.add_meta(self.clf_prob_feat_name, clf_prob)

        return corpus

    def summarize(self, corpus: Corpus, use_selector=True):
        objId_clf_prob = []

        for obj in corpus.iter_objs(self.obj_type, self.selector if use_selector else lambda _: True):
            objId_clf_prob.append((obj.id, obj.meta[self.clf_feat_name], obj.meta[self.clf_prob_feat_name]))

        return pd.DataFrame(list(objId_clf_prob),
                           columns=['id', self.clf_feat_name, self.clf_prob_feat_name]).set_index('id').sort_index(self.clf_prob_feat_name)

    def get_vocabulary(self):
        return self.vectorizer.vocabulary_

    def get_coefs(self, feature_names: List[str] = None, coef_func=None):
        """
        Get dataframe of classifier coefficients. By default, assumes it is a pipeline with a logistic regression component
        :param feature_names: list of feature names to get coefficients for
        :param coef_func: function for accessing the list of coefficients from the classifier model
        :return: DataFrame of features and coefficients, indexed by feature names
        """
        if feature_names is None:
            feature_names = self.vectorizer.vocabulary_

        if coef_func is None:
            coefs = self.clf.named_steps['logreg'].coef_[0].tolist()
        else:
            coefs = coef_func(self.clf)

        assert len(feature_names) == len(coefs)
        feats_coefs = sorted(list(zip(feature_names, coefs)), key=lambda x: x[1], reverse=True)
        return pd.DataFrame(feats_coefs, columns=['feat_name', 'coef']).set_index('feat_name')




