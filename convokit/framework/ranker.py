from convokit.model import Corpus, Conversation, User, Utterance
from typing import List, Callable, Union
from .framework import Framework


class Ranker(Framework):
    def __init__(self, obj_type: str,
                 score_func: Callable[[Union[User, Utterance, Conversation]], Union[int, float]],
                 filter_func: Callable[[Union[User, Utterance, Conversation]], bool] = None):
        self.obj_type = obj_type
        self.score_func = score_func
        self.filter_func = filter_func

    def evaluate(self, corpus: Corpus = None, objs: List[Union[User, Utterance, Conversation]] = None):
        assert (corpus is None and objs is not None) or (corpus is not None and objs is None)

        if objs is None:
            obj_iters = {"conversation": corpus.iter_conversations,
                         "user": corpus.iter_users,
                         "utterance": corpus.iter_utterances}
            return [(obj.id, self.score_func(obj) )for obj in obj_iters[self.obj_type]()
                     if self.filter_func is None or self.filter_func(obj)]
        else:
            return [(obj.id, self.score_func(obj)) for obj in objs]