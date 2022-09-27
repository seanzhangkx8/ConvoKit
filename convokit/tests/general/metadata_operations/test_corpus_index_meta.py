import unittest

from convokit.model import Utterance, Speaker, Corpus
from convokit.tests.general.metadata_operations.corpus_index_meta_helpers import get_basic_corpus
from convokit.tests.test_utils import reload_corpus_in_db_mode


class CorpusIndexMeta(unittest.TestCase):
    def basic_functions(self):
        """
        Test basic meta functions
        """

        first_utt = self.corpus.get_utterance("0")
        first_utt.meta["hey"] = 9

        # correct class type stored
        self.assertEqual(self.corpus.meta_index.utterances_index["hey"], [repr(type(9))])

        # keyErrors result in None output
        self.assertRaises(KeyError, lambda: first_utt.meta["nonexistent key"])

        # test that setting a custom get still works
        self.assertEqual(first_utt.meta.get("nonexistent_key", {}), {})

    def key_insertion_deletion(self):
        self.corpus.get_utterance("0").meta["foo"] = "bar"
        self.corpus.get_utterance("1").meta["foo"] = "bar2"
        self.corpus.get_utterance("2").meta["hey"] = "jude"

        self.corpus.get_conversation("convo_id_0").meta["convo_meta"] = 1
        self.corpus.get_speaker("alice").meta["surname"] = 1.0

        self.assertEqual(self.corpus.meta_index.utterances_index["foo"], [str(type("bar"))])
        self.assertEqual(self.corpus.meta_index.conversations_index["convo_meta"], [str(type(1))])
        self.assertEqual(self.corpus.meta_index.speakers_index["surname"], [str(type(1.0))])

        # test that deleting an attribute from an individual utterance fails to remove it
        del self.corpus.get_utterance("2").meta["hey"]
        self.corpus.get_utterance("2").meta["hey"]

        # test that delete_metadata works
        self.corpus.delete_metadata("utterance", "foo")
        self.assertRaises(KeyError, lambda: self.corpus.meta_index.utterances_index["foo"])
        self.assertRaises(KeyError, lambda: self.corpus.get_utterance("0").meta["foo"])

    def corpus_merge_add(self):
        self.corpus.get_utterance("0").meta["foo"] = "bar"
        self.corpus.get_utterance("1").meta["foo"] = "bar2"
        self.corpus.get_utterance("2").meta["hey"] = "jude"

        # test that adding separately initialized utterances with new metadata updates Index
        new_utt = Utterance(
            id="4",
            text="hello world",
            speaker=Speaker(id="alice", meta={"donkey": "kong"}),
            meta={"new": "meta"},
        )

        new_corpus = self.corpus.add_utterances([new_utt])
        self.assertTrue("new" in new_corpus.meta_index.utterances_index)
        self.assertTrue("donkey" in new_corpus.meta_index.speakers_index)

    def corpus_dump(self):
        self.corpus.get_utterance("0").meta["foo"] = "bar"
        self.corpus.get_utterance("1").meta["foo"] = "bar2"
        self.corpus.get_utterance("2").meta["hey"] = "jude"

        self.corpus.get_conversation("convo_id_0").meta["convo_meta"] = 1

        self.corpus.get_speaker("alice").meta["surname"] = 1.0
        self.corpus.dump("test_index_meta_corpus", base_path=".")
        corpus2 = Corpus(filename="test_index_meta_corpus")

        self.assertEqual(
            self.corpus.meta_index.utterances_index, corpus2.meta_index.utterances_index
        )
        self.assertEqual(self.corpus.meta_index.speakers_index, corpus2.meta_index.speakers_index)
        self.assertEqual(
            self.corpus.meta_index.conversations_index, corpus2.meta_index.conversations_index
        )
        self.assertEqual(self.corpus.meta_index.overall_index, corpus2.meta_index.overall_index)

    def multiple_types(self):
        self.corpus.get_utterance("2").meta["hey"] = None
        self.assertEqual(self.corpus.meta_index.utterances_index.get("hey", None), None)
        self.corpus.get_utterance("0").meta["hey"] = 5
        self.assertEqual(self.corpus.meta_index.utterances_index["hey"], [str(type(5))])
        self.corpus.get_utterance("1").meta["hey"] = "five"
        self.assertEqual(
            self.corpus.meta_index.utterances_index["hey"], [str(type(5)), str(type("five"))]
        )


class TestWithMem(CorpusIndexMeta):
    def setUp(self) -> None:
        self.corpus = get_basic_corpus()

    def test_basic_functions(self):
        self.basic_functions()

    def test_key_insertion_deletion(self):
        self.key_insertion_deletion()

    def test_corpus_merge_add(self):
        self.corpus_merge_add()

    def test_corpus_dump(self):
        self.corpus_dump()

    def test_multiple_types(self):
        self.multiple_types()


class TestWithDB(CorpusIndexMeta):
    def setUp(self) -> None:
        self.corpus = reload_corpus_in_db_mode(get_basic_corpus())

    def test_basic_functions(self):
        self.basic_functions()

    def test_key_insertion_deletion(self):
        self.key_insertion_deletion()

    def test_corpus_merge_add(self):
        self.corpus_merge_add()

    def test_corpus_dump(self):
        self.corpus_dump()

    def test_multiple_types(self):
        self.multiple_types()


if __name__ == "__main__":
    unittest.main()
