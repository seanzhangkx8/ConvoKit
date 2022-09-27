import unittest

from convokit import Conversation
from convokit.tests.general.fill_missing_convo_ids.fill_missing_convo_ids_helpers import (
    get_new_utterances_without_convo_ids,
    get_new_utterances_without_existing_convo_ids,
    construct_missing_convo_ids_corpus,
)
from convokit.tests.test_utils import reload_corpus_in_db_mode


class FillMissingConvoIds(unittest.TestCase):
    def corpus_with_utts_missing_convo_ids(self):
        convo_ids = [convo.id for convo in self.corpus.iter_conversations()]
        self.assertIn(Conversation.generate_default_conversation_id("0"), convo_ids)
        self.assertIn(Conversation.generate_default_conversation_id("3"), convo_ids)
        self.assertEqual(len(convo_ids), 2)

    def add_utts_without_convo_ids(self):
        self.corpus.add_utterances(get_new_utterances_without_convo_ids())
        num_convos = len(list(self.corpus.iter_conversations()))
        self.assertEqual(num_convos, 4)
        convo_ids = {convo.id for convo in self.corpus.iter_conversations()}
        root_utt_ids = [utt.id for utt in self.corpus.iter_utterances() if utt.reply_to is None]
        expected_convo_ids = {
            Conversation.generate_default_conversation_id(root_utt_id)
            for root_utt_id in root_utt_ids
        }
        self.assertEqual(convo_ids, expected_convo_ids)

    def add_utts_without_existing_convo_ids(self):
        self.corpus.add_utterances(get_new_utterances_without_existing_convo_ids())
        num_convos = len(list(self.corpus.iter_conversations()))
        self.assertEqual(num_convos, 2)
        convo_ids = {convo.id for convo in self.corpus.iter_conversations()}
        root_utt_ids = [utt.id for utt in self.corpus.iter_utterances() if utt.reply_to is None]
        expected_convo_ids = {
            Conversation.generate_default_conversation_id(root_utt_id)
            for root_utt_id in root_utt_ids
        }
        self.assertEqual(convo_ids, expected_convo_ids)


class TestWithDB(FillMissingConvoIds):
    def setUp(self) -> None:
        self.corpus = reload_corpus_in_db_mode(construct_missing_convo_ids_corpus())

    def test_corpus_with_utts_missing_convo_ids(self):
        self.corpus_with_utts_missing_convo_ids()

    def test_add_utts_with_missing_convo_ids(self):
        self.add_utts_without_convo_ids()

    def test_add_utts_without_existing_convo_ids(self):
        self.add_utts_without_existing_convo_ids()


class TestWithMem(FillMissingConvoIds):
    def setUp(self) -> None:
        self.corpus = construct_missing_convo_ids_corpus()

    def test_corpus_with_utts_missing_convo_ids(self):
        self.corpus_with_utts_missing_convo_ids()

    def test_add_utts_with_missing_convo_ids(self):
        self.add_utts_without_convo_ids()

    def test_add_utts_without_existing_convo_ids(self):
        self.add_utts_without_existing_convo_ids()


if __name__ == "__main__":
    unittest.main()
