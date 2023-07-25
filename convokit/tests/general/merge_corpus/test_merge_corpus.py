import unittest

from convokit.model import Utterance, Speaker, Corpus
from convokit.tests.general.merge_corpus.merge_corpus_helpers import (
    construct_base_corpus,
    construct_base_corpus_with_convo_id,
    construct_non_overlapping_corpus,
    construct_overlapping_corpus,
    construct_overlapping_corpus_with_convo_id,
)
from convokit.tests.test_utils import reload_corpus_in_db_mode


class CorpusMerge(unittest.TestCase):
    def no_overlap(self):
        """
        Basic merge: no overlap in utterance id
        """

        all_utt_ids = set(self.base_corpus.get_utterance_ids()) | set(
            self.non_overlapping_corpus.get_utterance_ids()
        )
        all_speaker_ids = set(self.base_corpus.get_speaker_ids()) | set(
            self.non_overlapping_corpus.get_speaker_ids()
        )

        merged = Corpus.merge(self.base_corpus, self.non_overlapping_corpus)
        self.assertEqual(len(list(merged.iter_utterances())), 6)
        self.assertEqual(len(list(merged.iter_speakers())), 6)

        for utt_id in all_utt_ids:
            self.assertTrue(merged.backend_mapper.has_data_for_component("utterance", utt_id))
        for speaker_id in all_speaker_ids:
            self.assertTrue(merged.backend_mapper.has_data_for_component("speaker", speaker_id))

        for component_type in self.base_corpus.backend_mapper.data.keys():
            self.assertEqual(self.base_corpus.backend_mapper.count_entries(component_type), 0)
        for component_type in self.non_overlapping_corpus.backend_mapper.data.keys():
            self.assertEqual(
                self.non_overlapping_corpus.backend_mapper.count_entries(component_type), 0
            )

    def with_overlap(self):
        """
        Basic merge: with overlap in utterance id (but utterance has same data & metadata)
        """

        all_utt_ids = set(self.base_corpus.get_utterance_ids()) | set(
            self.overlapping_corpus.get_utterance_ids()
        )
        all_speaker_ids = set(self.base_corpus.get_speaker_ids()) | set(
            self.overlapping_corpus.get_speaker_ids()
        )

        merged = Corpus.merge(self.base_corpus, self.overlapping_corpus)
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_speakers())), 5)

        for utt_id in all_utt_ids:
            self.assertTrue(merged.backend_mapper.has_data_for_component("utterance", utt_id))
        for speaker_id in all_speaker_ids:
            self.assertTrue(merged.backend_mapper.has_data_for_component("speaker", speaker_id))

        for component_type in self.base_corpus.backend_mapper.data.keys():
            self.assertEqual(self.base_corpus.backend_mapper.count_entries(component_type), 0)
        for component_type in self.overlapping_corpus.backend_mapper.data.keys():
            self.assertEqual(
                self.overlapping_corpus.backend_mapper.count_entries(component_type), 0
            )

    def overlap_diff_data(self):
        """
        Merge with overlap in utterance id and utterance has diff data but same metadata

        Warning should be printed. Original utterance data should be preserved.
        """
        all_utt_ids = set(self.base_corpus.get_utterance_ids()) | set(
            self.overlapping_corpus.get_utterance_ids()
        )
        all_speaker_ids = set(self.base_corpus.get_speaker_ids()) | set(
            self.overlapping_corpus.get_speaker_ids()
        )

        merged = Corpus.merge(self.base_corpus, self.overlapping_corpus)
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_speakers())), 5)
        self.assertEqual(len(list(self.base_corpus.iter_utterances())), 3)
        self.assertEqual(len(list(self.overlapping_corpus.iter_utterances())), 3)

        self.assertEqual(merged.get_utterance("2").text, "this is a test")
        self.assertEqual(merged.get_utterance("2").speaker.id, "charlie")

        for utt_id in all_utt_ids:
            self.assertTrue(merged.backend_mapper.has_data_for_component("utterance", utt_id))
        for speaker_id in all_speaker_ids:
            if (
                speaker_id == "candace"
            ):  # this speaker shouldn't be present due to overlap prioritization
                self.assertFalse(
                    merged.backend_mapper.has_data_for_component("speaker", speaker_id)
                )
            else:
                self.assertTrue(merged.backend_mapper.has_data_for_component("speaker", speaker_id))

        for component_type in self.base_corpus.backend_mapper.data.keys():
            self.assertEqual(self.base_corpus.backend_mapper.count_entries(component_type), 0)
        for component_type in self.overlapping_corpus.backend_mapper.data.keys():
            self.assertEqual(
                self.overlapping_corpus.backend_mapper.count_entries(component_type), 0
            )

    def overlap_diff_metadata(self):
        """
        Merge with overlap in utterance id and utterance has same data but diff metadata

        Second corpus utterance metadata should override if the keys are the same.
        """
        self.base_corpus.get_utterance("2").add_meta("hey", "jude")
        self.base_corpus.get_utterance("2").add_meta("the", "beatles")

        self.overlapping_corpus.get_utterance("2").add_meta("hey", "jude")
        self.overlapping_corpus.get_utterance("2").add_meta("the", "ringo")
        self.overlapping_corpus.get_utterance("2").add_meta("let it", "be")

        all_utt_ids = set(self.base_corpus.get_utterance_ids()) | set(
            self.overlapping_corpus.get_utterance_ids()
        )
        all_speaker_ids = set(self.base_corpus.get_speaker_ids()) | set(
            self.overlapping_corpus.get_speaker_ids()
        )

        merged = Corpus.merge(self.base_corpus, self.overlapping_corpus)
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_speakers())), 5)

        self.assertEqual(len(merged.get_utterance("2").meta), 3)
        self.assertEqual(merged.get_utterance("2").meta["the"], "ringo")

        for utt_id in all_utt_ids:
            self.assertTrue(merged.backend_mapper.has_data_for_component("utterance", utt_id))
            self.assertTrue(
                merged.backend_mapper.has_data_for_component("meta", f"utterance_{utt_id}")
            )
        for speaker_id in all_speaker_ids:
            self.assertTrue(merged.backend_mapper.has_data_for_component("speaker", speaker_id))
            self.assertTrue(
                merged.backend_mapper.has_data_for_component("meta", f"speaker_{speaker_id}")
            )

        for component_type in self.base_corpus.backend_mapper.data.keys():
            self.assertEqual(self.base_corpus.backend_mapper.count_entries(component_type), 0)
        for component_type in self.overlapping_corpus.backend_mapper.data.keys():
            self.assertEqual(
                self.overlapping_corpus.backend_mapper.count_entries(component_type), 0
            )

    def overlap_convo_metadata(self):
        """
        Merge with overlap in conversation with metadata differences.

        Expect second corpus convo metadata to override if keys are the same
        """

        self.base_corpus_with_convo_id.get_conversation("convo1").add_meta("hey", "jude")
        self.base_corpus_with_convo_id.get_conversation("convo1").add_meta("hello", "world")

        self.overlapping_corpus_with_convo_id.get_conversation("convo1").add_meta("hey", "jude")
        self.overlapping_corpus_with_convo_id.get_conversation("convo1").add_meta("hello", "food")
        self.overlapping_corpus_with_convo_id.get_conversation("convo1").add_meta("what", "a mood")

        merged = Corpus.merge(self.base_corpus_with_convo_id, self.overlapping_corpus_with_convo_id)
        self.assertEqual(len(merged.get_conversation("convo1").meta), 3)
        self.assertEqual(merged.get_conversation("convo1").meta["hello"], "food")

        self.assertTrue(merged.backend_mapper.has_data_for_component("conversation", "convo1"))
        self.assertTrue(merged.backend_mapper.has_data_for_component("meta", "conversation_convo1"))

        self.assertFalse(
            self.base_corpus_with_convo_id.backend_mapper.has_data_for_component(
                "conversation", "convo1"
            )
        )
        self.assertFalse(
            self.overlapping_corpus_with_convo_id.backend_mapper.has_data_for_component(
                "conversation", "convo1"
            )
        )
        self.assertFalse(
            self.base_corpus_with_convo_id.backend_mapper.has_data_for_component(
                "meta", "conversation_convo1"
            )
        )
        self.assertFalse(
            self.overlapping_corpus_with_convo_id.backend_mapper.has_data_for_component(
                "meta", "conversation_convo1"
            )
        )

    def corpus_metadata(self):
        """
        Merge with overlap in corpus metadata

        Expect second corpus metadata to override if keys are the same
        """
        self.base_corpus.add_meta("politeness", 0.95)
        self.base_corpus.add_meta("toxicity", 0.8)

        self.non_overlapping_corpus.add_meta("toxicity", 0.9)
        self.non_overlapping_corpus.add_meta("paggro", 1.0)

        merged = Corpus.merge(self.base_corpus, self.non_overlapping_corpus)
        self.assertEqual(len(merged.meta), 3)
        self.assertEqual(merged.meta["toxicity"], 0.9)

    def add_utterance(self):
        self.base_corpus.get_utterance("2").add_meta("hey", "jude")
        self.base_corpus.get_utterance("2").add_meta("hello", "world")

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
        added = self.base_corpus.add_utterances(utts)

        self.assertIs(added, self.base_corpus)
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
            self.assertFalse(hasattr(utt, "_temp_backend"))


class TestWithMem(CorpusMerge):
    def setUp(self) -> None:
        self.base_corpus = construct_base_corpus()
        self.base_corpus_with_convo_id = construct_base_corpus_with_convo_id()
        self.non_overlapping_corpus = construct_non_overlapping_corpus()
        self.overlapping_corpus = construct_overlapping_corpus()
        self.overlapping_corpus_with_convo_id = construct_overlapping_corpus_with_convo_id()

    def test_no_overlap(self):
        self.no_overlap()

    def test_with_overlap(self):
        self.with_overlap()

    def test_overlap_diff_data(self):
        self.overlap_diff_data()

    def test_overlap_diff_metadata(self):
        self.overlap_diff_metadata()

    def test_overlap_convo_metadata(self):
        self.overlap_convo_metadata()

    def test_corpus_metadata(self):
        self.corpus_metadata()

    def test_add_utterance(self):
        self.add_utterance()


class TestWithDB(CorpusMerge):
    def setUp(self) -> None:
        self.base_corpus = reload_corpus_in_db_mode(construct_base_corpus())
        self.base_corpus_with_convo_id = reload_corpus_in_db_mode(
            construct_base_corpus_with_convo_id()
        )
        self.non_overlapping_corpus = reload_corpus_in_db_mode(construct_non_overlapping_corpus())
        self.overlapping_corpus = reload_corpus_in_db_mode(construct_overlapping_corpus())
        self.overlapping_corpus_with_convo_id = reload_corpus_in_db_mode(
            construct_overlapping_corpus_with_convo_id()
        )

    def test_no_overlap(self):
        self.no_overlap()

    def test_with_overlap(self):
        self.with_overlap()

    def test_overlap_diff_data(self):
        self.overlap_diff_data()

    def test_overlap_diff_metadata(self):
        self.overlap_diff_metadata()

    def test_overlap_convo_metadata(self):
        self.overlap_convo_metadata()

    def test_corpus_metadata(self):
        self.corpus_metadata()

    def test_add_utterance(self):
        self.add_utterance()


if __name__ == "__main__":
    unittest.main()
