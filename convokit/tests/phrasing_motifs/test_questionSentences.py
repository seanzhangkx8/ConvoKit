import unittest

from convokit.phrasing_motifs.questionSentences import QuestionSentences
from convokit.tests.test_utils import small_burr_corpus_parsed, reload_corpus_in_db_mode


def parsed_burr_sir_corpus_with_lowercase_are():
    corpus = small_burr_corpus_parsed()
    for utterance in corpus.iter_utterances():
        parsed = utterance.retrieve_meta("parsed")
        for sentence in parsed:
            if sentence["toks"][0]["tok"] == "Are":
                sentence["toks"][0]["tok"] = "are"
    return corpus


class TestQuestionSentences(unittest.TestCase):
    def use_caps(self):
        transformer = QuestionSentences(
            input_field="sentences", output_field="questions", use_caps=True
        )
        transformed_corpus = transformer.transform(self.corpus)

        expected_sentences_list = [[], ["Who's asking?"]]
        for utterance, expected_sentences in zip(
            transformed_corpus.iter_utterances(), expected_sentences_list
        ):
            self.assertListEqual(expected_sentences, list(utterance.retrieve_meta("questions")))

    def use_caps_false(self):
        transformer = QuestionSentences(
            input_field="sentences", output_field="questions", use_caps=False
        )
        transformed_corpus = transformer.transform(self.corpus)

        expected_sentences_list = [["Are you Aaron Burr, sir?"], ["Who's asking?"]]
        for utterance, expected_sentences in zip(
            transformed_corpus.iter_utterances(), expected_sentences_list
        ):
            self.assertListEqual(expected_sentences, list(utterance.retrieve_meta("questions")))


class TestWithMem(TestQuestionSentences):
    def setUp(self) -> None:
        self.corpus = parsed_burr_sir_corpus_with_lowercase_are()

    def test_use_caps(self):
        self.use_caps()

    def test_use_caps_false(self):
        self.use_caps_false()


class TestWithDB(TestQuestionSentences):
    def setUp(self) -> None:
        self.corpus = reload_corpus_in_db_mode(parsed_burr_sir_corpus_with_lowercase_are())

    def test_use_caps(self):
        self.use_caps()

    def test_use_caps_false(self):
        self.use_caps_false()
