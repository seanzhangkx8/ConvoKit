import unittest
from unittest import TestCase

from scipy.sparse import coo_matrix

from convokit import BoWTransformer
from convokit.tests.test_utils import (
    small_burr_corpus,
    BURR_SIR_TEXT_1,
    BURR_SIR_TEXT_2,
    reload_corpus_in_db_mode,
)


def burr_sir_sentence_1_vector():
    return coo_matrix([[1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1]]).tocsr()


def burr_sir_sentence_2_vector():
    return coo_matrix([[0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0]]).tocsr()


def assert_sparse_matrices_equal(matrix1, matrix2):
    assert matrix1.shape == matrix2.shape
    assert (matrix1 != matrix2).nnz == 0


class FakeVectorizer:
    def transform(self, texts):
        assert len(texts) == 2
        assert texts[0] == BURR_SIR_TEXT_1
        assert texts[1] == BURR_SIR_TEXT_2

        return coo_matrix(
            [[1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0]]
        ).tocsr()

    def fit(self, docs):
        pass


class TestBoWTransformer(TestCase):
    def transform_utterances(self):
        transformer = BoWTransformer(obj_type="utterance", vectorizer=FakeVectorizer())
        corpus = transformer.fit_transform(self.corpus)

        expected_vectors = [burr_sir_sentence_1_vector(), burr_sir_sentence_2_vector()]

        for expected_vector, utterance in zip(expected_vectors, corpus.iter_utterances()):
            actual_vector = utterance.get_vector("bow_vector")
            assert_sparse_matrices_equal(expected_vector, actual_vector)


class TestWithDB(TestBoWTransformer):
    def setUp(self) -> None:
        self.corpus = reload_corpus_in_db_mode(small_burr_corpus())

    def test_transform_utterances(self):
        self.transform_utterances()


class TestWithMem(TestBoWTransformer):
    def setUp(self) -> None:
        self.corpus = small_burr_corpus()

    def test_transform_utterances(self):
        self.transform_utterances()


if __name__ == "__main__":
    unittest.main()
