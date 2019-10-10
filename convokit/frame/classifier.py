from .framework import Framework
from .util import extract_convo_features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Union
from pandas import DataFrame
from convokit.model import Corpus, Conversation
from collections import defaultdict
from random import shuffle, choice
from scipy.sparse import csr_matrix

class Classifier():
    def __init__(self):
        pass

    def fit(self, corpus: Corpus):
        pass

    def evaluate(self, corpus: Corpus):
        pass

    def fit_evaluate(self, corpus: Corpus):
        pass