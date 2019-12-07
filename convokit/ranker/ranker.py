from convokit.model import Corpus, Conversation, User, Utterance
from typing import List, Callable, Union
from convokit import Transformer
import pandas as pd

class Ranker(Transformer):
    def __init__(self, obj_type: str,
                 score_func: Callable[[Union[User, Utterance, Conversation]], Union[int, float]],
                 selector: Callable[[Union[User, Utterance, Conversation]], bool] = lambda obj: True,
                 score_feat_name: str = "ranking_score", rank_feat_name: str = "rank"):
        """

        :param obj_type: type of Corpus object to rank: 'conversation', 'user', or 'utterance'
        :param score_func: function for computing the score of a given object
        :param selector: function to select for Corpus objects to transform
        :param score_feat_name: metadata feature name to use in annotation for score value, default: "ranking_score"
        :param rank_feat_name: metadata feature name to use in annotation for the rank value, default: "rank"
        """
        self.obj_type = obj_type
        self.score_func = score_func
        self.score_feat_name = score_feat_name
        self.rank_feat_name = rank_feat_name
        self.selector = selector

    def transform(self, corpus: Corpus) -> Corpus:
        """
        Annotate scores and rankings on corpus objects
        :param corpus:
        :return:
        """
        id_to_score_rank_df = self.summarize(corpus)

        for obj in corpus.iter_objs(obj_type=self.obj_type):
            if obj.id in id_to_score_rank_df.index:
                obj.add_meta(self.score_feat_name, id_to_score_rank_df.loc[obj.id][self.score_feat_name])
                obj.add_meta(self.rank_feat_name, id_to_score_rank_df.loc[obj.id][self.rank_feat_name])
            else:
                obj.add_meta(self.score_feat_name, None)
                obj.add_meta(self.rank_feat_name, None)
        return corpus

    def summarize(self, corpus: Corpus = None, objs: List[Union[User, Utterance, Conversation]] = None):
        """

        :param corpus:
        :param objs:
        :return:
        """
        if ((corpus is None) and (objs is None)) or ((corpus is not None) and (objs is not None)):
            raise ValueError("summarize() takes in either a Corpus or a list of users / utterances / conversations")

        if objs is None:
            obj_iters = {"conversation": corpus.iter_conversations,
                         "user": corpus.iter_users,
                         "utterance": corpus.iter_utterances}
            obj_scores = [(obj.id, self.score_func(obj)) for obj in obj_iters[self.obj_type](self.selector)]
        else:
            obj_scores = [(obj.id, self.score_func(obj)) for obj in objs]

        df = pd.DataFrame(obj_scores, columns=["id", self.score_feat_name])\
                        .set_index('id').sort_values(self.score_feat_name, ascending=False)
        df[self.rank_feat_name] = [idx+1 for idx, _ in enumerate(df.index)]

        return df