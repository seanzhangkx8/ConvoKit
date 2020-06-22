from convokit import Corpus, CorpusComponent
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from typing import Callable, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
from .classifier import Classifier
import numpy as np
from .util import extract_vector_feats_and_label


class VectorClassifier(Classifier):
    """
    Transformer that trains a classifier on the Corpus components' text vector representation
    (e.g. bag-of-words, TF-IDF, etc)

    Corpus must have a vector of the specified `vector_name`.

    Inherits from `Classifier` and has access to its methods. # TODO check this

    :param obj_type: "speaker", "utterance", or "conversation"
    :param vector_name: the metadata key where the Corpus object text vector is stored
    :param labeller: a (lambda) function that takes a Corpus object and returns True (y=1) or False (y=0) - i.e. labeller defines the y value of the object for fitting
    :param clf: a sklearn Classifier. By default, clf is a Pipeline with StandardScaler and LogisticRegression
    :param clf_feat_name: the metadata key to store the classifier prediction value under; default: "prediction"
    :param clf_prob_feat_name: the metadata key to store the classifier prediction score under; default: "pred_score"
    """
    def __init__(self, obj_type: str, labeller: Callable[[CorpusComponent], bool] = lambda x: True,
                 clf=None, clf_feat_name: str = "prediction", clf_prob_feat_name: str = "pred_score"):
        if clf is None:
            print("Initializing default classification model (standard scaled logistic regression)")
            clf = Pipeline([("standardScaler", StandardScaler(with_mean=False)),
                            ("logreg", LogisticRegression(solver='liblinear'))])

        super().__init__(obj_type=obj_type, pred_feats=[], labeller=labeller,
                         clf=clf, clf_feat_name=clf_feat_name, clf_prob_feat_name=clf_prob_feat_name)

    def fit(self, corpus: Corpus, vector_name: str, columns: List[str] = None, y=None,
            selector: Callable[[CorpusComponent], bool] = lambda x: True):
        """
        Fit the Transformer's internal classifier model on the vector matrix named `vector_name` that represents one of
        the Corpus components, with an optional selector that filters for objects to be fit on.

        :param corpus: the target Corpus
        :param vector_name: name of vector matrix
        :param columns: list of column names of vector matrix to use; uses all columns by default.
        :param selector: a (lambda) function that takes a Corpus object and returns True or False
            (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :return: the fitted VectorClassifier
        """
        # collect texts for vectorization
        obj_ids = []
        y = []
        for obj in corpus.iter_objs(self.obj_type, selector):
            obj_ids.append(obj.id)
            y.append(self.labeller(obj))
        X = corpus.get_vector_matrix(vector_name).get_vectors(obj_ids, columns)
        self.clf.fit(X, y)
        return self

    def transform(self, corpus: Corpus, vector_name: str, columns: List[str] = None,
                  selector: Callable[[CorpusComponent], bool] = lambda x: True) -> Corpus:
        """
        Annotate the corpus components with the classifier prediction and prediction score, with an optional selector
        that filters for objects to be classified. Objects that are not selected will get a metadata value of 'None'
        instead of the classifier prediction.

        :param corpus: the target Corpus
        :param vector_name: name of vector matrix
        :param columns: list of column names of vector matrix to use; uses all columns by default.
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.

        :return: the target Corpus annotated
        """
        objs = []
        for obj in corpus.iter_objs(self.obj_type):
            if selector(obj):
                objs.append(obj)
            else:
                obj.add_meta(self.clf_feat_name, None)
                obj.add_meta(self.clf_prob_feat_name, None)

        obj_ids = [obj.id for obj in objs]
        X = corpus.get_vector_matrix(vector_name).get_vectors(obj_ids, columns)

        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            obj = objs[idx]
            obj.add_meta(self.clf_feat_name, clf)
            obj.add_meta(self.clf_prob_feat_name, clf_prob)
        return corpus

    def transform_objs(self, objs: List[CorpusComponent]) -> List[CorpusComponent]:
        """
        Run classifier on list of Corpus component objects and annotate them with their predictions and
        prediction scores.

        :param objs: list of Corpus objects
        :return: list of annotated Corpus objects
        """
        X, _ = extract_vector_feats_and_label(corpus=None, objs=objs, obj_type=None, vector_name=self.vector_name,
                                              labeller=self.labeller, selector=None)

        X = X.toarray()
        # obj_ids = [obj.id for obj in objs]
        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            obj = objs[idx]
            obj.add_meta(self.clf_feat_name, clf)
            obj.add_meta(self.clf_prob_feat_name, clf_prob)

        return objs

    def fit_transform(self, corpus: Corpus, vector_name: str, columns: List[str] = None, y=None,
                      selector: Callable[[CorpusComponent], bool] = lambda x: True) -> Corpus:
        self.fit(corpus, vector_name, columns, selector=selector)
        return self.transform(corpus, vector_name, columns, selector=selector)

    def summarize(self, corpus: Corpus, selector: Callable[[CorpusComponent], bool] = lambda x: True):
        """
        Generate a DataFrame indexed by object id with the classifier predictions and scores.

        :param corpus: the annotated Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :return: a pandas DataFrame
        """
        objId_clf_prob = []

        for obj in corpus.iter_objs(self.obj_type, selector):
            objId_clf_prob.append((obj.id, obj.meta[self.clf_feat_name], obj.meta[self.clf_prob_feat_name]))

        return pd.DataFrame(list(objId_clf_prob),
                            columns=['id', self.clf_feat_name, self.clf_prob_feat_name])\
                            .set_index('id').sort_values(self.clf_prob_feat_name, ascending=False)

    def summarize_objs(self, objs: List[CorpusComponent]):
        """
        Generate a pandas DataFrame (indexed by object id, with prediction and prediction score columns) of classification results.

        Runs on a list of Corpus objects.

        :param objs: list of Corpus objects
        :return: pandas DataFrame indexed by Corpus object id
        """
        objId_clf_prob = []
        for obj in objs:
            objId_clf_prob.append((obj.id, obj.meta[self.clf_feat_name], obj.meta[self.clf_prob_feat_name]))

        return pd.DataFrame(list(objId_clf_prob),
                            columns=['id', self.clf_feat_name, self.clf_prob_feat_name]).set_index('id').sort_values(
                            self.clf_prob_feat_name)


    def evaluate_with_train_test_split(self, corpus: Corpus, vector_name: str, columns: List[str] = None,
                                       selector: Callable[[CorpusComponent], bool] = lambda x: True,
                                       test_size: float = 0.2):
        """
        Evaluate the performance of predictive features (Classifier.pred_feats) in predicting for the label,
        using a train-test split.

        Run either on a Corpus (with Classifier labeller, selector, obj_type settings) or a list of Corpus objects

        :param corpus: target Corpus
        :param selector: a (lambda) function that takes a Corpus object and returns True or False (i.e. include /
            exclude). By default, the selector includes all objects of the specified type in the Corpus.
        :param test_size: size of test set
        :return: accuracy and confusion matrix
        """
        X, y = extract_vector_feats_and_label(corpus, self.obj_type, vector_name, columns, self.labeller, selector)

        print("Running a train-test-split evaluation...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        accuracy = np.mean(preds == y_test)
        print("Done.")
        return accuracy, confusion_matrix(y_true=y_test, y_pred=preds)

    def evaluate_with_cv(self, corpus: Corpus, vector_name: str, columns: List[str] = None,
                         cv=KFold(n_splits=5),
                         selector: Callable[[CorpusComponent], bool] = lambda x: True
                         ):
        """
        Evaluate the performance of predictive features (Classifier.pred_feats) in predicting for the label,
        using cross-validation for data splitting.

        Run either on a Corpus (with Classifier labeller, selector, obj_type settings) or a list of Corpus objects.

        :param corpus: target Corpus
        :param objs: target list of Corpus objects (do not pass in corpus if using this)
        :param cv: cross-validation model to use: KFold(n_splits=5) by default.
        :param selector: if running on a Corpus, this is a (lambda) function that takes a Corpus object and returns True or False (i.e. include / exclude). By default, the selector includes all objects of the specified type in the Corpus.

        :return: cross-validated accuracy score
        """

        X, y = extract_vector_feats_and_label(corpus, self.obj_type, vector_name, columns, self.labeller, selector)
        print("Running a cross-validated evaluation...")
        score = cross_val_score(self.clf, X, y, cv=cv)
        print("Done.")
        return score

    def transform_objs(self, objs: List[CorpusComponent]) -> List[CorpusComponent]:
        raise NotImplementedError("transform_objs() is not supported for VectorClassifier")


