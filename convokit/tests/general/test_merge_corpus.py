import unittest
from convokit.model import Utterance, Speaker, Corpus


class CorpusMerge(unittest.TestCase):
    def test_no_overlap(self):
        """
        Basic merge: no overlap in utterance id
        """
        corpus1 = Corpus(
            utterances=[
                Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
                Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
                Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
            ]
        )

        corpus2 = Corpus(
            utterances=[
                Utterance(id="3", text="i like pie", speaker=Speaker(id="delta")),
                Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
                Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
            ]
        )

        all_utt_ids = set(corpus1.get_utterance_ids()) | set(corpus2.get_utterance_ids())
        all_speaker_ids = set(corpus1.get_speaker_ids()) | set(corpus2.get_speaker_ids())

        merged = Corpus.merge(corpus1, corpus2)
        self.assertEqual(len(list(merged.iter_utterances())), 6)
        self.assertEqual(len(list(merged.iter_speakers())), 6)

        for utt_id in all_utt_ids:
            self.assertTrue(utt_id in merged.storage.get_collection_ids("utterance"))
        for speaker_id in all_speaker_ids:
            self.assertTrue(speaker_id in merged.storage.get_collection_ids("speaker"))

        for collection in corpus1.storage.data.values():
            self.assertEqual(len(collection), 0)
        for collection in corpus2.storage.data.values():
            self.assertEqual(len(collection), 0)

    def test_with_overlap(self):
        """
        Basic merge: with overlap in utterance id (but utterance has same data & metadata)
        """
        corpus1 = Corpus(
            utterances=[
                Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
                Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
                Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
            ]
        )

        corpus2 = Corpus(
            utterances=[
                Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
                Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
                Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
            ]
        )

        all_utt_ids = set(corpus1.get_utterance_ids()) | set(corpus2.get_utterance_ids())
        all_speaker_ids = set(corpus1.get_speaker_ids()) | set(corpus2.get_speaker_ids())

        merged = Corpus.merge(corpus1, corpus2)
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_speakers())), 5)

        for utt_id in all_utt_ids:
            self.assertTrue(utt_id in merged.storage.get_collection_ids("utterance"))
        for speaker_id in all_speaker_ids:
            self.assertTrue(speaker_id in merged.storage.get_collection_ids("speaker"))

        for collection in corpus1.storage.data.values():
            self.assertEqual(len(collection), 0)
        for collection in corpus2.storage.data.values():
            self.assertEqual(len(collection), 0)

    def test_overlap_diff_data(self):
        """
        Merge with overlap in utterance id and utterance has diff data but same metadata

        Warning should be printed. Original utterance data should be preserved.
        """
        corpus1 = Corpus(
            utterances=[
                Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
                Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
                Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
            ]
        )

        corpus2 = Corpus(
            utterances=[
                Utterance(id="2", text="this is a test2", speaker=Speaker(id="candace")),
                Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
                Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
            ]
        )

        all_utt_ids = set(corpus1.get_utterance_ids()) | set(corpus2.get_utterance_ids())
        all_speaker_ids = set(corpus1.get_speaker_ids()) | set(corpus2.get_speaker_ids())

        merged = Corpus.merge(corpus1, corpus2)
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_speakers())), 5)
        self.assertEqual(len(list(corpus1.iter_utterances())), 3)
        self.assertEqual(len(list(corpus2.iter_utterances())), 3)

        self.assertEqual(merged.get_utterance("2").text, "this is a test")
        self.assertEqual(merged.get_utterance("2").speaker.id, "charlie")

        for utt_id in all_utt_ids:
            self.assertTrue(utt_id in merged.storage.get_collection_ids("utterance"))
        for speaker_id in all_speaker_ids:
            if (
                speaker_id == "candace"
            ):  # this speaker shouldn't be present due to overlap prioritization
                self.assertFalse(speaker_id in merged.storage.get_collection_ids("speaker"))
            else:
                self.assertTrue(speaker_id in merged.storage.get_collection_ids("speaker"))

        for collection in corpus1.storage.data.values():
            self.assertEqual(len(collection), 0)
        for collection in corpus2.storage.data.values():
            self.assertEqual(len(collection), 0)

    def test_overlap_diff_metadata(self):
        """
        Merge with overlap in utterance id and utterance has same data but diff metadata

        Second corpus utterance metadata should override if the keys are the same.
        """
        corpus1 = Corpus(
            utterances=[
                Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
                Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
                Utterance(
                    id="2",
                    text="this is a test",
                    speaker=Speaker(id="charlie"),
                    meta={"hey": "jude", "the": "beatles"},
                ),
            ]
        )

        corpus2 = Corpus(
            utterances=[
                Utterance(
                    id="2",
                    text="this is a test",
                    speaker=Speaker(id="charlie"),
                    meta={"hey": "jude", "the": "ringo", "let it": "be"},
                ),
                Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
                Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
            ]
        )

        all_utt_ids = set(corpus1.get_utterance_ids()) | set(corpus2.get_utterance_ids())
        all_speaker_ids = set(corpus1.get_speaker_ids()) | set(corpus2.get_speaker_ids())

        merged = Corpus.merge(corpus1, corpus2)
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_speakers())), 5)

        self.assertEqual(len(merged.get_utterance("2").meta), 3)
        self.assertEqual(merged.get_utterance("2").meta["the"], "ringo")

        for utt_id in all_utt_ids:
            self.assertTrue(utt_id in merged.storage.get_collection_ids("utterance"))
            self.assertTrue(f"utterance_{utt_id}" in merged.storage.get_collection_ids("meta"))
        for speaker_id in all_speaker_ids:
            self.assertTrue(speaker_id in merged.storage.get_collection_ids("speaker"))
            self.assertTrue(f"speaker_{speaker_id}" in merged.storage.get_collection_ids("meta"))

        for collection in corpus1.storage.data.values():
            self.assertEqual(len(collection), 0)
        for collection in corpus2.storage.data.values():
            self.assertEqual(len(collection), 0)

    def test_overlap_convo_metadata(self):
        """
        Merge with overlap in conversation with metadata differences.

        Expect second corpus convo metadata to override if keys are the same
        """
        corpus1 = Corpus(
            utterances=[
                Utterance(
                    id="0",
                    conversation_id="convo1",
                    text="hello world",
                    speaker=Speaker(id="alice"),
                ),
                Utterance(
                    id="1",
                    conversation_id="convo1",
                    text="my name is bob",
                    speaker=Speaker(id="bob"),
                ),
                Utterance(
                    id="2",
                    conversation_id="convo1",
                    text="this is a test",
                    speaker=Speaker(id="charlie"),
                ),
            ]
        )

        corpus2 = Corpus(
            utterances=[
                Utterance(
                    id="2",
                    conversation_id="convo1",
                    text="this is a test",
                    speaker=Speaker(id="charlie"),
                ),
                Utterance(
                    id="4",
                    conversation_id="convo1",
                    text="this is a sentence",
                    speaker=Speaker(id="echo"),
                ),
                Utterance(
                    id="5", conversation_id="convo1", text="goodbye", speaker=Speaker(id="foxtrot")
                ),
            ]
        )

        corpus1.get_conversation("convo1").add_meta("hey", "jude")
        corpus1.get_conversation("convo1").add_meta("hello", "world")

        corpus2.get_conversation("convo1").add_meta("hey", "jude")
        corpus2.get_conversation("convo1").add_meta("hello", "food")
        corpus2.get_conversation("convo1").add_meta("what", "a mood")

        merged = Corpus.merge(corpus1, corpus2)
        self.assertEqual(len(merged.get_conversation("convo1").meta), 3)
        self.assertEqual(merged.get_conversation("convo1").meta["hello"], "food")

        self.assertTrue("convo1" in merged.storage.get_collection_ids("conversation"))
        self.assertTrue("conversation_convo1" in merged.storage.get_collection_ids("meta"))

        self.assertFalse("convo1" in corpus1.storage.get_collection_ids("conversation"))
        self.assertFalse("convo1" in corpus2.storage.get_collection_ids("conversation"))
        self.assertFalse("conversation_convo1" in corpus1.storage.get_collection_ids("meta"))
        self.assertFalse("conversation_convo1" in corpus2.storage.get_collection_ids("meta"))

    def test_corpus_metadata(self):
        """
        Merge with overlap in corpus metadata

        Expect second corpus metadata to override if keys are the same
        """
        corpus1 = Corpus(
            utterances=[
                Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
                Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
                Utterance(id="2", text="this is a test", speaker=Speaker(id="charlie")),
            ]
        )

        corpus2 = Corpus(
            utterances=[
                Utterance(id="3", text="i like pie", speaker=Speaker(id="delta")),
                Utterance(id="4", text="this is a sentence", speaker=Speaker(id="echo")),
                Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
            ]
        )

        corpus1.add_meta("politeness", 0.95)
        corpus1.add_meta("toxicity", 0.8)

        corpus2.add_meta("toxicity", 0.9)
        corpus2.add_meta("paggro", 1.0)

        merged = Corpus.merge(corpus1, corpus2)
        self.assertEqual(len(merged.meta), 3)
        self.assertEqual(merged.meta["toxicity"], 0.9)

    def test_add_utterance(self):
        corpus1 = Corpus(
            utterances=[
                Utterance(id="0", text="hello world", speaker=Speaker(id="alice")),
                Utterance(id="1", text="my name is bob", speaker=Speaker(id="bob")),
                Utterance(
                    id="2",
                    text="this is a test",
                    speaker=Speaker(id="charlie"),
                    meta={"hey": "jude", "hello": "world"},
                ),
            ]
        )

        utts = [
            Utterance(id="1", text="i like pie", speaker=Speaker(id="delta")),
            Utterance(
                id="2",
                text="this is a test",
                speaker=Speaker(id="charlie"),
                meta={"hello": "food", "what": "a mood"},
            ),
            Utterance(id="5", text="goodbye", speaker=Speaker(id="foxtrot")),
        ]
        added = corpus1.add_utterances(utts)

        self.assertIs(added, corpus1)
        self.assertEqual(len(list(added.iter_utterances())), 4)
        self.assertEqual(set(added.get_utterance_ids()), {"0", "1", "2", "5"})
        self.assertEqual(set(added.get_speaker_ids()), {"alice", "bob", "charlie", "foxtrot"})
        self.assertEqual(len(added.get_utterance("2").meta), 3)
        self.assertEqual(added.get_utterance("2").meta["hello"], "food")
        self.assertIn("what", added.get_utterance("2").meta)
        self.assertEqual(added.get_utterance("1").text, "my name is bob")
        self.assertEqual(added.get_utterance("1").speaker.id, "bob")
        self.assertEqual(added.get_utterance("5").text, "goodbye")
        self.assertEqual(added.get_utterance("5").speaker.id, "foxtrot")

        for utt in added.iter_utterances():
            self.assertFalse(hasattr(utt, "_temp_storage"))


if __name__ == "__main__":
    unittest.main()
