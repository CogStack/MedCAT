import os

from medcat.vocab import Vocab

import unittest


class RegressionModelVocabTests(unittest.TestCase):
    VOCAB_DATA_PATH = os.path.join(os.path.dirname(__file__),
                              'creation', 'vocab_data.txt')

    @classmethod
    def setUpClass(cls):
        cls.vocab = Vocab()
        cls.vocab.add_words(cls.VOCAB_DATA_PATH)

    def test_has_same_vector_lengths(self):
        all_lengths = set()
        for w in self.vocab.vec_index2word.values():
            all_lengths.add(len(self.vocab.vec(w)))
        self.assertEqual(len(all_lengths), 1, f"Expected equal lengths. Got: {all_lengths}")

    def test_all_words_have_vectors(self):
        for w in self.vocab.vocab:
            with self.subTest(f"Word: {repr(w)}"):
                # NOTE: if not there, will raise an exception
                self.assertIsNotNone(self.vocab.vec(w))


if __name__ == '__main__':
    unittest.main()
