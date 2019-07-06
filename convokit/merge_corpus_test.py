import unittest
from model import Corpus, Utterance, User


class CorpusMerge(unittest.TestCase):
    def test_no_overlap(self):
        """
        Basic merge: no overlap in utterance id
        """
        corpus1 = Corpus(utterances = [
            Utterance(id=0, text="hello world", user=User(name="alice")),
            Utterance(id=1, text="my name is bob", user=User(name="bob")),
            Utterance(id=2, text="this is a test", user=User(name="charlie")),
        ])

        corpus2 = Corpus(utterances = [
            Utterance(id=3, text="i like pie", user=User(name="delta")),
            Utterance(id=4, text="this is a sentence", user=User(name="echo")),
            Utterance(id=5, text="goodbye", user=User(name="foxtrot")),
        ])

        merged = corpus1.merge_corpus(corpus2)
        self.assertEqual(len(list(merged.iter_utterances())), 6)
        self.assertEqual(len(list(merged.iter_users())), 6)
        self.assertEqual(len(list(corpus1.iter_utterances())), 3)
        self.assertEqual(len(list(corpus2.iter_utterances())), 3)

    def test_with_overlap(self):
        """
        Basic merge: with overlap in utterance id (but utterance has same data & metadata)
        """
        corpus1 = Corpus(utterances = [
            Utterance(id=0, text="hello world", user=User(name="alice")),
            Utterance(id=1, text="my name is bob", user=User(name="bob")),
            Utterance(id=2, text="this is a test", user=User(name="charlie")),
        ])

        corpus2 = Corpus(utterances = [
            Utterance(id=2, text="this is a test", user=User(name="charlie")),
            Utterance(id=4, text="this is a sentence", user=User(name="echo")),
            Utterance(id=5, text="goodbye", user=User(name="foxtrot")),
        ])

        merged = corpus1.merge_corpus(corpus2)
        self.assertEqual
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_users())), 5)
        self.assertEqual(len(list(corpus1.iter_utterances())), 3)
        self.assertEqual(len(list(corpus2.iter_utterances())), 3)

    def test_overlap_diff_data(self):
        """
        Merge with overlap in utterance id and utterance has diff data but same metadata

        Warning should be printed
        """
        corpus1 = Corpus(utterances = [
            Utterance(id=0, text="hello world", user=User(name="alice")),
            Utterance(id=1, text="my name is bob", user=User(name="bob")),
            Utterance(id=2, text="this is a test", user=User(name="charlie")),
        ])

        corpus2 = Corpus(utterances = [
            Utterance(id=2, text="this is a test2", user=User(name="candace")),
            Utterance(id=4, text="this is a sentence", user=User(name="echo")),
            Utterance(id=5, text="goodbye", user=User(name="foxtrot")),
        ])

        merged = corpus1.merge_corpus(corpus2)
        self.assertEqual(len(list(merged.iter_utterances())), 5)
        self.assertEqual(len(list(merged.iter_users())), 5)
        self.assertEqual(len(list(corpus1.iter_utterances())), 3)
        self.assertEqual(len(list(corpus2.iter_utterances())), 3)

        self.assertEqual(merged.get_utterance(2).text, "this is a test2")
        self.assertEqual(merged.get_utterance(2).user, User(name="candace"))


if __name__ == '__main__':
    unittest.main()
