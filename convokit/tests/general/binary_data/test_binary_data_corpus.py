import shutil
import unittest

from convokit import Corpus
from convokit.tests.general.binary_data.binary_data_helpers import construct_corpus_with_binary_data
from convokit.tests.test_utils import reload_corpus_in_db_mode

DUMPED_CORPUS_NAME = "binary_corpus_test"


class CorpusBinaryData(unittest.TestCase):
    def dump_and_load_with_binary(self):
        """
        Dump a corpus containing speakers with binary metadata and utterances with binary metadata
        Check that dumped corpus is successfully loaded with the same data
        """

        alice = self.corpus.get_speaker("alice")
        bob = self.corpus.get_speaker("bob")

        self.corpus.dump(DUMPED_CORPUS_NAME, "./")
        corpus2 = Corpus(filename=DUMPED_CORPUS_NAME)

        alice2 = corpus2.get_speaker("alice")
        bob2 = corpus2.get_speaker("bob")

        self.assertEqual(alice.meta, alice2.meta)
        self.assertEqual(self.corpus.get_utterance("0").meta, corpus2.get_utterance("0").meta)
        self.assertEqual(bob.meta, bob2.meta)
        self.assertEqual(self.corpus.get_utterance("1").meta, corpus2.get_utterance("1").meta)

    def partial_load_corpus(self):
        self.corpus.dump(DUMPED_CORPUS_NAME, "./")
        corpus2 = Corpus(
            filename=DUMPED_CORPUS_NAME, utterance_start_index=0, utterance_end_index=1
        )
        self.assertEqual(len(list(corpus2.iter_utterances())), 2)
        self.assertEqual(self.corpus.get_utterance("0"), corpus2.get_utterance("0"))
        self.assertEqual(self.corpus.get_utterance("1"), corpus2.get_utterance("1"))

    def partial_load_start_idx_specified_only(self):
        self.corpus.dump(DUMPED_CORPUS_NAME, "./")

        corpus2 = Corpus(filename=DUMPED_CORPUS_NAME, utterance_start_index=1)

        self.assertEqual(len(list(corpus2.iter_utterances())), 2)
        self.assertEqual(self.corpus.get_utterance("1"), corpus2.get_utterance("1"))
        self.assertEqual(self.corpus.get_utterance("2"), corpus2.get_utterance("2"))

    def partial_load_end_idx_specified_only(self):
        self.corpus.dump(DUMPED_CORPUS_NAME, "./")
        corpus2 = Corpus(filename=DUMPED_CORPUS_NAME, utterance_end_index=0)
        self.assertEqual(len(list(corpus2.iter_utterances())), 1)
        self.assertEqual(self.corpus.get_utterance("0"), corpus2.get_utterance("0"))

    def partial_load_invalid_start_index(self):
        self.corpus.dump(DUMPED_CORPUS_NAME, "./")
        corpus2 = Corpus(filename=DUMPED_CORPUS_NAME, utterance_start_index=99)
        self.assertEqual(len(list(corpus2.iter_utterances())), 0)

    def partial_load_invalid_end_index(self):
        self.corpus.dump(DUMPED_CORPUS_NAME, "./")
        corpus2 = Corpus(filename=DUMPED_CORPUS_NAME, utterance_end_index=-1)
        self.assertEqual(len(list(corpus2.iter_utterances())), 0)

    def tearDown(self) -> None:
        shutil.rmtree(DUMPED_CORPUS_NAME)


class TestWithDB(CorpusBinaryData):
    def setUp(self) -> None:
        self.corpus = reload_corpus_in_db_mode(construct_corpus_with_binary_data())

    def test_dump_and_load_with_binary(self):
        self.dump_and_load_with_binary()

    def test_partial_load_corpus(self):
        self.partial_load_corpus()

    def test_partial_load_start_idx_specified_only(self):
        self.partial_load_start_idx_specified_only()

    def test_partial_load_end_idx_specified_only(self):
        self.partial_load_end_idx_specified_only()

    def test_partial_load_invalid_start_index(self):
        self.partial_load_invalid_start_index()

    def test_partial_load_invalid_end_index(self):
        self.partial_load_invalid_end_index()


class TestWithMem(CorpusBinaryData):
    def setUp(self) -> None:
        self.corpus = construct_corpus_with_binary_data()

    def test_dump_and_load_with_binary(self):
        self.dump_and_load_with_binary()

    def test_partial_load_corpus(self):
        self.partial_load_corpus()

    def test_partial_load_start_idx_specified_only(self):
        self.partial_load_start_idx_specified_only()

    def test_partial_load_end_idx_specified_only(self):
        self.partial_load_end_idx_specified_only()

    def test_partial_load_invalid_start_index(self):
        self.partial_load_invalid_start_index()

    def test_partial_load_invalid_end_index(self):
        self.partial_load_invalid_end_index()


if __name__ == "__main__":
    unittest.main()
