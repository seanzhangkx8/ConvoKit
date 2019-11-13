import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as CV
import string
from convokit import Transformer
from convokit.model import Corpus, Utterance
from typing import List, Callable
from matplotlib import pyplot as plt

exclude = set(string.punctuation)

class FightingWords(Transformer):
    """
    Adapted from: https://github.com/jmhessel/FightingWords
    
    # DESIGN DECISION: restrict unit to utterance list #TODO

    - ngram; an int describing up to what n gram you want to consider (1 is unigrams,
        2 is bigrams + unigrams, etc). Ignored if a custom CountVectorizer is passed.
    - prior; either a float describing a uniform prior, or a vector describing a prior
        over vocabulary items. If you're using a predefined vocabulary, make sure to specify that
        when you make your CountVectorizer object.
    - threshold: the z-score threshold for annotating utterances with identified ngrams
    - top_k: use the top k ngrams when annotating utterances
    - annot_method: "top_k" or "threshold" to specify which annotation method to use
    - cv; a sklearn.feature_extraction.text.CountVectorizer object, if desired.
    """
    def __init__(self, l1_selector: Callable[[Utterance], bool],
                 l2_selector: Callable[[Utterance], bool],
                 ngram=1, prior=0.1, threshold=1, top_k=10, annot_method="top_k", cv=None):
        self.l1_selector = l1_selector
        self.l2_selector = l2_selector
        self.ngram = ngram
        self.prior = prior
        self.cv = cv
        self.threshold = threshold
        self.top_k = top_k
        assert annot_method in ["top_k", "threshold"]
        self.annot_method = annot_method
        self.ngram_zscores = None
        self._count_matrix = None
        if self.cv is None and type(self.prior) is not float:
            raise ValueError("If using a non-uniform prior, you must pass a count vectorizer with "
                             "the vocabulary parameter set.")
        if self.cv is None:
            self.cv = CV(decode_error='ignore', min_df=10, max_df=.5, ngram_range=(1, 3),
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
        - A dict of length |Vocab| with (n-gram, zscore) pairs.'''
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
        self._count_matrix = count_matrix
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
        return {index_to_term[i]: z_scores[i] for i in sorted_indices}

    def fit(self, corpus: Corpus):
        l1, l2 = [], []
        for utt in corpus.iter_utterances():
            if self.l1_selector(utt):
                l1.append(utt)
            elif self.l2_selector(utt):
                l2.append(utt)

        if len(l1) == 0:
            raise ValueError("l1_func returned 0 valid utterances.")
        if len(l2) == 0:
            raise ValueError("l2_func returned 0 valid utterances.")

        print("l1_func returned {} valid utterances. l2_func returned {} valid utterances.".format(len(l1), len(l2)))

        self.ngram_zscores = self._bayes_compare_language(l1, l2)
        print("ngram zscores computed.")

    def get_ngram_zscores(self):
        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")
        return list(self.ngram_zscores.items())

    def top_k_ngrams(self):
        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")
        ngram_zscores_list = self.get_ngram_zscores()
        top_k_l1 = list(reversed([x[0] for x in ngram_zscores_list[-self.top_k:]]))
        top_k_l2 = [x[0] for x in ngram_zscores_list[:self.top_k]]
        return top_k_l1, top_k_l2

    def ngrams_past_threshold(self):
        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")
        l1_ngrams = []
        l2_ngrams = []
        for ngram, zscore in self.ngram_zscores.items():
            if zscore > self.threshold:
                l1_ngrams.append(ngram)
            elif zscore < -self.threshold:
                l2_ngrams.append(ngram)
        return l1_ngrams, l2_ngrams

    def transform(self, corpus: Corpus) -> Corpus:
        l1_ngrams, l2_ngrams = self.top_k_ngrams() if self.annot_method == "top_k" else self.ngrams_past_threshold()

        for utt in corpus.iter_utterances(): # improve the efficiency of this; tricky because ngrams #TODO
            utt.meta['fighting_words_l1'] = [ngram for ngram in l1_ngrams if ngram in utt.text]
            utt.meta['fighting_words_l2'] = [ngram for ngram in l2_ngrams if ngram in utt.text]
        return corpus

    def get_zscore(self, ngram):
        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")
        return self.ngram_zscores.get(ngram, None)

    def analyze(self, corpus: Corpus):
        if self.ngram_zscores is None:
            self.fit(corpus)
        self.plot_fighting_words()

    def plot_fighting_words(self):
        # Adapted from https://gist.github.com/xandaschofield/3c4070b2f232b185ce6a09e47b4e7473
        if self.ngram_zscores is None:
            raise ValueError("fit() must be run on a corpus first.")

        x_vals = self._count_matrix.sum(axis=0)
        y_vals = [x[1] for x in self.get_ngram_zscores()]
        sizes = abs(np.array(y_vals)) * 1.5
        neg_color, pos_color, insig_color = ('orange', 'purple', 'grey')
        colors = []
        annots = []

        l1_sig_ngrams, l2_sig_ngrams = self.top_k_ngrams() if self.annot_method == "top_k" else self.ngrams_past_threshold()
        l1_sig_ngrams = set(l1_sig_ngrams)
        l2_sig_ngrams = set(l2_sig_ngrams)

        for i, (term, y_val) in enumerate(self.get_ngram_zscores()):
            if term in l1_sig_ngrams:
                colors.append(pos_color)
                annots.append(term)
            elif term in l2_sig_ngrams:
                colors.append(neg_color)
                annots.append(term)
            else:
                colors.append(insig_color)
                annots.append(None)

        fig, ax = plt.subplots()
        ax.scatter(x_vals, y_vals, c=colors, s=sizes, linewidth=0)
        for i, annot in enumerate(annots):
            if annot is not None:
                ax.annotate(annot, (x_vals[i], y_vals[i]), color=colors[i], size=sizes[i])
        ax.set_xscale('log')
        ax.set_title("Weighted log-odds ratio against Frequency of word within topic")
        plt.show()




