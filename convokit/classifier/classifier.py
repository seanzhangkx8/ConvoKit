from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn import svm
from convokit.classifier.util import *
from convokit import Transformer


class Classifier(Transformer):
    def __init__(self, obj_type: str, pred_feats: List[str],
                 labeller: Callable[[Union[User, Utterance, Conversation]], bool] = lambda x: True,
                 selector: Callable[[Union[User, Utterance, Conversation]], bool] = lambda x: True,
                 clf=None, clf_feat_name: str = "prediction", clf_prob_feat_name: str = "score"):
        self.pred_feats = pred_feats
        self.labeller = labeller
        self.selector = selector
        self.obj_type = obj_type
        self.clf = svm.SVC(C=0.02, kernel='linear', probability=True) if clf is None else clf
        self.clf_feat_name = clf_feat_name
        self.clf_prob_feat_name = clf_prob_feat_name


    def fit(self, corpus: Corpus):
        X, y = extract_feats_and_label(corpus, self.obj_type, self.pred_feats, self.labeller, self.selector)
        self.clf.fit(X, y)

    def transform(self, corpus: Corpus) -> Corpus:
        obj_id_to_feats = extract_feats_dict(corpus, self.obj_type, self.pred_feats, self.selector)
        feats_df = pd.DataFrame.from_dict(obj_id_to_feats, orient='index')
        X = csr_matrix(feats_df.values)
        idx_to_id = {idx: obj_id for idx, obj_id in enumerate(list(obj_id_to_feats))}
        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)

        for idx, (clf, clf_prob) in enumerate(list(zip(clfs, clfs_probs))):
            corpus_obj = corpus.get_object(self.obj_type, idx_to_id[idx])
            corpus_obj.add_meta(self.clf_feat_name, clf)
            corpus_obj.add_meta(self.clf_prob_feat_name, clf_prob)

        return corpus

    def analyze(self, corpus: Corpus = None, objs: List[Union[User, Utterance, Conversation]] = None):
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError("analyze() takes in either a Corpus or a list of users / utterances / conversations")

        if corpus:
            X = extract_feats(corpus, self.obj_type, self.pred_feats, self.selector)
            obj_ids = [obj.id for obj in corpus.iter_objs(self.obj_type, self.selector)]
        else:
            assert objs is not None
            X = np.array([list(extract_feats_from_obj(obj, self.pred_feats).values()) for obj in objs])
            obj_ids = [obj.id for obj in objs]

        clfs, clfs_probs = self.clf.predict(X), self.clf.predict_proba(X)[:, 1]

        return pd.DataFrame(list(zip(obj_ids, clfs, clfs_probs)),
                            columns=['id', self.clf_feat_name, self.clf_prob_feat_name]).set_index('id')


    def evaluate_with_train_test_split(self, corpus: Corpus = None,
                 objs: List[Union[User, Utterance, Conversation]] = None,
                 test_size: float = 0.2):
        """

        :param corpus:
        :param objs:
        :param test_size:
        :return: accuracy and confusion matrix
        """
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError("analyze() takes in either a Corpus or a list of users / utterances / conversations")

        if corpus:
            print("Using corpus objects...")
            X, y = extract_feats_and_label(corpus, self.obj_type, self.pred_feats, self.labeller, self.selector)
            obj_ids = [obj.id for obj in corpus.iter_objs(self.obj_type, self.selector)]
        else:
            print("Using input list of corpus objects...")
            X = np.array([list(extract_feats_from_obj(obj, self.pred_feats).values()) for obj in objs])
            y = np.array([self.labeller(obj) for obj in objs])
            obj_ids = [obj.id for obj in objs]

        print("Running a train-test-split evaluation...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        accuracy = np.mean(preds == y_test)
        return accuracy, confusion_matrix(y_true=y, y_pred=preds)

    def evaluate_with_cv(self, corpus: Corpus = None,
                         objs: List[Union[User, Utterance, Conversation]] = None, cv=KFold(n_splits=5)):
        """

        :param corpus:
        :param objs:
        :param cv:
        :return:
        """
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError("analyze() takes in either a Corpus or a list of users / utterances / conversations")

        if corpus:
            print("Using corpus objects...")
            X, y = extract_feats_and_label(corpus, self.obj_type, self.pred_feats, self.labeller, self.selector)
            obj_ids = [obj.id for obj in corpus.iter_objs(self.obj_type, self.selector)]
        else:
            print("Using input list of corpus objects...")
            X = np.array([list(extract_feats_from_obj(obj, self.pred_feats).values()) for obj in objs])
            y = np.array([self.labeller(obj) for obj in objs])
            obj_ids = [obj.id for obj in objs]

        print("Running a cross-validated evaluation...")
        return cross_val_score(self.clf, X, y, cv=cv)
