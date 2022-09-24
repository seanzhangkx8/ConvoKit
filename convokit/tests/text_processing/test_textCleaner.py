import unittest

from convokit.tests.test_utils import small_burr_corpus, reload_corpus_in_db_mode
from convokit.text_processing.textCleaner import clean_str, TextCleaner


class TestTextCleaner(unittest.TestCase):
    def no_options(self):
        cleaner = TextCleaner(text_cleaner=lambda text: "cleaned text")
        cleaned_corpus = cleaner.transform(self.corpus)

        for utterance in cleaned_corpus.iter_utterances():
            self.assertEqual(utterance.text, "cleaned text")

    def no_text_replacement(self):
        cleaner = TextCleaner(text_cleaner=lambda text: "cleaned text", replace_text=False)
        cleaned_corpus = cleaner.transform(self.corpus_copy)

        for original_utterance, cleaned_utterance in zip(
            self.corpus.iter_utterances(), cleaned_corpus.iter_utterances()
        ):
            self.assertEqual(original_utterance.text, cleaned_utterance.text)
            self.assertEqual(cleaned_utterance.meta["cleaned"], "cleaned text")

    def save_original(self):
        cleaner = TextCleaner(
            text_cleaner=lambda text: "cleaned text", replace_text=True, save_original=True
        )
        cleaned_corpus = cleaner.transform(self.corpus_copy)

        for original_utterance, cleaned_utterance in zip(
            self.corpus.iter_utterances(), cleaned_corpus.iter_utterances()
        ):
            self.assertEqual(cleaned_utterance.text, "cleaned text")
            self.assertEqual(original_utterance.text, cleaned_utterance.meta["original"])

    def test_clean_str_replacements(self):
        original_str = "https://mywebsite.com myemail@gmail.com (123) 456-7890 1,000 $"
        cleaned_str = clean_str(original_str)
        self.assertEqual("<url> <email> <phone> <number> <cur>", cleaned_str)

    def test_clean_str_text(self):
        original_str = "BA OBAMA ĐỘI BA Ô BA OBAMA"  # vietnamese joke https://twitter.com/tetracarbon/status/735009694515789824
        cleaned_str = clean_str(original_str)
        self.assertEqual("ba obama doi ba o ba obama", cleaned_str)


class TestWithMem(TestTextCleaner):
    def setUp(self) -> None:
        self.corpus = small_burr_corpus()
        self.corpus_copy = small_burr_corpus()

    def test_no_options(self):
        self.no_options()

    def test_no_text_replacement(self):
        self.no_text_replacement()

    def test_save_original(self):
        self.save_original()


class TestWithDB(TestTextCleaner):
    def setUp(self) -> None:
        self.corpus = reload_corpus_in_db_mode(small_burr_corpus())
        self.corpus_copy = reload_corpus_in_db_mode(small_burr_corpus())

    def test_no_options(self):
        self.no_options()

    def test_no_text_replacement(self):
        self.no_text_replacement()

    def test_save_original(self):
        self.save_original()
