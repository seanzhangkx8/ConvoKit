import unittest
from convokit.model import Utterance, Speaker, Corpus
from convokit import download

class CorpusLoading(unittest.TestCase):
    """
    Load a variety of existing (small) corpora to verify that there are no backward compatibility issues
    """

    def test_load_subreddit(self):
        corpus = Corpus(download('subreddit-hey'))

    def test_load_tennis(self):
        corpus = Corpus(download('tennis-corpus'))

    def test_load_politeness(self):
        corpus = Corpus(download('wikipedia-politeness-corpus'))

    def test_load_switchboard(self):
        corpus = Corpus(download("switchboard-corpus"))

    def test_convos_gone_awry(self):
        corpus = Corpus(download('conversations-gone-awry-corpus'))

if __name__ == '__main__':
    unittest.main()
