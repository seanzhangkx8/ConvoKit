import unittest

import pytest

from convokit import download
from convokit.model import Corpus

"""
Load a variety of existing (small) corpora to verify that there are no backward compatibility issues
"""


@pytest.mark.parametrize("storage_type", ["mem", "db"])
def test_load_dump_subreddit(storage_type):
    corpus = Corpus(download("subreddit-hey"), storage_type=storage_type)
    corpus.dump("subreddit")


@pytest.mark.parametrize("storage_type", ["mem", "db"])
def test_load_dump_tennis(storage_type):
    corpus = Corpus(download("tennis-corpus"), storage_type=storage_type)
    corpus.dump("tennis-corpus")


@pytest.mark.parametrize("storage_type", ["mem", "db"])
def test_load_dump_politeness(storage_type):
    corpus = Corpus(download("wikipedia-politeness-corpus"), storage_type=storage_type)
    corpus.dump("wikipedia-politeness-corpus")


@pytest.mark.parametrize("storage_type", ["mem", "db"])
def test_load_dump_switchboard(storage_type):
    corpus = Corpus(download("switchboard-corpus"), storage_type=storage_type)
    corpus.dump("switchboard-corpus")


@pytest.mark.parametrize("storage_type", ["mem", "db"])
def test_load_wikiconv(storage_type):
    corpus = Corpus(download("wikiconv-2004"), storage_type=storage_type)
    corpus.dump("switchboard-corpus")


if __name__ == "__main__":
    unittest.main()
