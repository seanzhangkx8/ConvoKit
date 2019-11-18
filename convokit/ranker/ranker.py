from convokit.model import Corpus, Conversation, User, Utterance
from typing import List, Callable, Union
from convokit import Transformer
import pandas as pd

class Ranker(Transformer):
    def __init__(self, obj_type: str,
                 score_func: Callable[[Union[User, Utterance, Conversation]], Union[int, float]],
                 selector: Callable[[Union[User, Utterance, Conversation]], bool] = lambda obj: True,
                 score_feat_name: str = "ranking_score"):
        self.obj_type = obj_type
        self.score_func = score_func
        self.score_feat_name = score_feat_name
        self.selector = selector

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Annotate ranking on corpus objects
        :param corpus:
        :return:
        """

        for obj in corpus.iter_objs(obj_type=self.obj_type):
            if self.selector(obj):
                obj.add_meta(self.score_feat_name, self.score_func(obj))
            else:
                obj.add_meta(self.score_feat_name, None)
        return corpus


    def analyze(self, corpus: Corpus = None, objs: List[Union[User, Utterance, Conversation]] = None):
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError("analyze() takes in either a Corpus or a list of users / utterances / conversations")

        if objs is None:
            obj_iters = {"conversation": corpus.iter_conversations,
                         "user": corpus.iter_users,
                         "utterance": corpus.iter_utterances}
            obj_scores = [(obj.id, self.score_func(obj)) for obj in obj_iters[self.obj_type](self.selector)]
        else:
            obj_scores = [(obj.id, self.score_func(obj)) for obj in objs]

        return pd.DataFrame(obj_scores, columns=["id", "score"]).set_index('id').sort_values('score', ascending=False)
