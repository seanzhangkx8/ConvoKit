import unittest

from convokit import download
from convokit.model import Corpus
from convokit.tests.test_utils import reload_corpus_in_db_mode


class CorpusFromPandas(unittest.TestCase):
    def reconstruction_stats(self):
        """
        Test that reconstructing the Corpus from outputted dataframes results in the same number of corpus components
        """
        assert len(self.new_corpus.speakers) == len(self.corpus.speakers)
        assert len(self.new_corpus.conversations) == len(self.corpus.conversations)
        assert len(self.new_corpus.utterances) == len(self.corpus.utterances)

    def _check_reconstruction_metadata(self, new_corpus):
        assert set(self.corpus.random_utterance().meta) == set(new_corpus.random_utterance().meta)
        assert set(self.corpus.random_conversation().meta) == set(
            new_corpus.random_conversation().meta
        )
        assert set(self.corpus.random_speaker().meta) == set(new_corpus.random_speaker().meta)

    def _check_convo_integrity(self, new_corpus):
        for convo in new_corpus.iter_conversations():
            assert convo.check_integrity(verbose=False)

    def full_reconstruction(self):
        self._check_reconstruction_metadata(self.new_corpus)
        self._check_convo_integrity(self.new_corpus)

    def no_speakers_df(self):
        test_corpus = Corpus.from_pandas(
            utterances_df=self.utt_df, speakers_df=None, conversations_df=self.convo_df
        )
        self._check_convo_integrity(test_corpus)

    def no_convos_df(self):
        test_corpus = Corpus.from_pandas(
            utterances_df=self.utt_df, speakers_df=self.speaker_df, conversations_df=None
        )
        self._check_convo_integrity(test_corpus)

    def no_speaker_convo_dfs(self):
        test_corpus = Corpus.from_pandas(utterances_df=self.utt_df)
        self._check_convo_integrity(test_corpus)


class TestWithMem(CorpusFromPandas):
    def setUp(self) -> None:
        self.corpus = Corpus(download("subreddit-hey"), storage_type="mem")
        self.utt_df = self.corpus.get_utterances_dataframe()
        self.convo_df = self.corpus.get_conversations_dataframe()
        self.speaker_df = self.corpus.get_speakers_dataframe()
        self.new_corpus = Corpus.from_pandas(self.utt_df, self.speaker_df, self.convo_df)

    def test_reconstruction_stats(self):
        self.reconstruction_stats()

    def test_full_reconstruction(self):
        self.full_reconstruction()

    def test_no_speakers_df(self):
        self.no_speakers_df()

    def test_no_convos_df(self):
        self.no_convos_df()

    def test_no_speaker_convo_dfs(self):
        self.no_speaker_convo_dfs()


class TestWithDB(CorpusFromPandas):
    def setUp(self) -> None:
        self.corpus = Corpus(download("subreddit-hey"), storage_type="db")
        self.utt_df = self.corpus.get_utterances_dataframe()
        self.convo_df = self.corpus.get_conversations_dataframe()
        self.speaker_df = self.corpus.get_speakers_dataframe()
        self.new_corpus = reload_corpus_in_db_mode(
            Corpus.from_pandas(self.utt_df, self.speaker_df, self.convo_df)
        )

    def test_reconstruction_stats(self):
        self.reconstruction_stats()

    def test_full_reconstruction(self):
        self.full_reconstruction()

    def test_no_speakers_df(self):
        self.no_speakers_df()

    def test_no_convos_df(self):
        self.no_convos_df()

    def test_no_speaker_convo_dfs(self):
        self.no_speaker_convo_dfs()


if __name__ == "__main__":
    unittest.main()
