import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as CV
import string
from .framework import Framework
from convokit.model import Corpus, Utterance
from typing import List, Callable

exclude = set(string.punctuation)

class FightingWords(Framework):
    """
    Adapted from: https://github.com/jmhessel/FightingWords
    
    # DESIGN DECISION: restrict unit to utterance list #TODO

    - ngram; an int describing up to what n gram you want to consider (1 is unigrams,
        2 is bigrams + unigrams, etc). Ignored if a custom CountVectorizer is passed.
    - prior; either a float describing a uniform prior, or a vector describing a prior
        over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
        when you make your CountVectorizer object.
    - cv; a sklearn.feature_extraction.text.CountVectorizer object, if desired.
    """
    def __init__(self, l1_func: Callable[[Utterance], bool],
                 l2_func: Callable[[Utterance], bool],
                 ngram=1, prior=0.1, cv=None):
        self.l1_func = l1_func
        self.l2_func = l2_func
        self.ngram = ngram
        self.prior = prior
        self.cv = cv

        if self.cv is None and type(self.prior) is not float:
            raise ValueError("If using a non-uniform prior, you must pass a count vectorizer with "
                             "the vocabulary parameter set.")
        if self.cv is None:
            self.cv = CV(decode_error='ignore', min_df=10, max_df=.5, ngram_range=(1, self.ngram),
                    binary=False,
                    max_features=15000)
    @staticmethod
    def _basic_sanitize(in_string):
        '''Returns a very roughly sanitized version of the input string.'''
        return_string = (b' '.join(in_string.encode('ascii', 'ignore').strip().split())).decode('ascii')
        return_string = ''.join(ch for ch in return_string if ch not in exclude)
        return_string = return_string.lower()
        return_string = ' '.join(return_string.split())
        return return_string

    def _bayes_compare_language(self, l1: List[Utterance], l2: List[Utterance]):
        '''
        Arguments:
        - l1, l2; a list of strings from each language sample

        Returns:
        - A list of length |Vocab| where each entry is a (n-gram, zscore) tuple.'''
        l1 = [FightingWords._basic_sanitize(utt.text) for utt in l1]
        l2 = [FightingWords._basic_sanitize(utt.text) for utt in l2]

        counts_mat = self.cv.fit_transform(l1+l2).toarray()
        # Now sum over languages...
        vocab_size = len(self.cv.vocabulary_)
        print("Vocab size is {}".format(vocab_size))
        if type(self.prior) is float:
            priors = np.array([self.prior for _ in range(vocab_size)])
        else:
            priors = self.prior
        z_scores = np.empty(priors.shape[0])
        count_matrix = np.empty([2, vocab_size], dtype=np.float32)
        count_matrix[0, :] = np.sum(counts_mat[:len(l1), :], axis=0)
        count_matrix[1, :] = np.sum(counts_mat[len(l1):, :], axis=0)
        a0 = np.sum(priors)
        n1 = 1.*np.sum(count_matrix[0, :])
        n2 = 1.*np.sum(count_matrix[1, :])
        print("Comparing language...")
        for i in range(vocab_size):
            #compute delta
            term1 = np.log((count_matrix[0, i] + priors[i])/(n1 + a0 - count_matrix[0, i] - priors[i]))
            term2 = np.log((count_matrix[1, i] + priors[i])/(n2 + a0 - count_matrix[1, i] - priors[i]))
            delta = term1 - term2
            #compute variance on delta
            var = 1./(count_matrix[0, i] + priors[i]) + 1./(count_matrix[1, i] + priors[i])
            #store final score
            z_scores[i] = delta/np.sqrt(var)
        index_to_term = {v: k for k, v in self.cv.vocabulary_.items()}
        sorted_indices = np.argsort(z_scores)
        return_list = []
        for i in sorted_indices:
            return_list.append((index_to_term[i], z_scores[i]))
        return return_list

    def evaluate(self, corpus: Corpus):
        l1, l2 = [], []
        for utt in corpus.iter_utterances():
            if self.l1_func(utt):
                l1.append(utt)
            elif self.l2_func(utt):
                l2.append(utt)

        if len(l1) == 0:
            raise ValueError("l1_func returned 0 valid utterances.")
        if len(l2) == 0:
            raise ValueError("l2_func returned 0 valid utterances.")

        print("l1_func returned {} valid utterances. l2_func returned {} valid utterances.".format(len(l1), len(l2)))

        return self._bayes_compare_language(l1, l2)






