import unittest

from convokit.tests.test_utils import small_burr_corpus_parsed, reload_corpus_in_db_mode
from convokit.text_processing.textToArcs import TextToArcs


class TestTextToArcs(unittest.TestCase):
    def default_options(self):
        transformer = TextToArcs(output_field="arcs")

        transformed_corpus = transformer.transform(self.corpus)
        expected_arcs = [
            [
                "me_* pardon>* pardon_* pardon_me",
                "aaron_* are>* are_* are_burr are_sir are_you burr_* burr_aaron sir_* you_*",
            ],
            [
                "depends_* depends_that that>* that_*",
                "'s_* asking_'s asking_* asking_who who>'s who>* who_*",
            ],
        ]
        for expected_arc, utterance in zip(expected_arcs, transformed_corpus.iter_utterances()):
            self.assertListEqual(expected_arc, utterance.meta["arcs"])

    def use_start_false(self):
        transformer = TextToArcs(output_field="arcs", use_start=False)
        transformed_corpus = transformer.transform(self.corpus)
        expected_arcs = [
            [
                "me_* pardon_* pardon_me",
                "aaron_* are_* are_burr are_sir are_you burr_* burr_aaron sir_* you_*",
            ],
            ["depends_* depends_that that_*", "'s_* asking_'s asking_* asking_who who_*"],
        ]
        for expected_arc, utterance in zip(expected_arcs, transformed_corpus.iter_utterances()):
            self.assertListEqual(expected_arc, utterance.meta["arcs"])

    def root_only(self):
        transformer = TextToArcs(output_field="arcs", root_only=True)
        transformed_corpus = transformer.transform(self.corpus)
        expected_arcs = [
            ["pardon>* pardon_* pardon_me", "are>* are_* are_burr are_sir are_you"],
            ["depends_* depends_that that>*", "asking_'s asking_* asking_who who>'s who>*"],
        ]
        for expected_arc, utterance in zip(expected_arcs, transformed_corpus.iter_utterances()):
            self.assertListEqual(expected_arc, utterance.meta["arcs"])


class TestWithDB(TestTextToArcs):
    def setUp(self) -> None:
        self.corpus = reload_corpus_in_db_mode(small_burr_corpus_parsed())

    def test_default_options(self):
        self.default_options()

    def test_use_start_false(self):
        self.use_start_false()

    def test_root_only(self):
        self.root_only()


class TestWithMem(TestTextToArcs):
    def setUp(self) -> None:
        self.corpus = small_burr_corpus_parsed()

    def test_default_options(self):
        self.default_options()

    def test_use_start_false(self):
        self.use_start_false()

    def test_root_only(self):
        self.root_only()
