from convokit.model import Corpus, Speaker, Conversation, FightingWords
from convokit.transformer import Transformer
from collections import Counter, Any
from scipy.stats import wilcoxon, mannwhitneyu
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import re
import random

class Balance(Transformer):
    def __init__(
        self,
        primary_threshold=0.50001,
        window_ps_threshold=0.6,
        window_size=2.5,
        sliding_size=30,
        cur_cut=0,
        remove_first_last_utt=True
    ):
        self.primary_threshold = primary_threshold
        self.window_ps_threshold = window_ps_threshold
        self.window_size = window_size
        self.sliding_size = sliding_size
        self.cur_cut = cur_cut
        self.remove_first_last_utt = remove_first_last_utt

    def fit(
        self,
        corpus: Corpus,
        groupA_func , # for each conversation, which speakers are in groupA and which speakers are in groupB
        groupB_func,
    ):
        pass
        

    def transform(
        self,
        corpus: Corpus
    ):
        pass