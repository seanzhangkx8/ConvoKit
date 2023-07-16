import unittest

import pytest

from convokit import download
from convokit.model import Corpus

"""
Load a variety of existing (small) corpora to verify that there are no backward compatibility issues
"""


@pytest.mark.parametrize("backend", ["mem", "db"])
def test_load_dump_subreddit(backend):
    corpus = Corpus(download("subreddit-hey"), backend=backend)
    corpus.dump("subreddit")


@pytest.mark.parametrize("backend", ["mem", "db"])
def test_load_dump_tennis(backend):
    corpus = Corpus(download("tennis-corpus"), backend=backend)
    corpus.dump("tennis-corpus")


@pytest.mark.parametrize("backend", ["mem", "db"])
def test_load_dump_politeness(backend):
    corpus = Corpus(download("wikipedia-politeness-corpus"), backend=backend)
    corpus.dump("wikipedia-politeness-corpus")


@pytest.mark.parametrize("backend", ["mem", "db"])
def test_load_dump_switchboard(backend):
    corpus = Corpus(download("switchboard-corpus"), backend=backend)
    corpus.dump("switchboard-corpus")


@pytest.mark.parametrize("backend", ["mem", "db"])
def test_load_wikiconv(backend):
    corpus = Corpus(download("wikiconv-2004"), backend=backend)
    corpus.dump("switchboard-corpus")


if __name__ == "__main__":
    unittest.main()
