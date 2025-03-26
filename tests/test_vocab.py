import os
import shutil
import unittest
from medcat.vocab import Vocab
from collections import Counter
import numpy as np


class CATTests(unittest.TestCase):

    def setUp(self) -> None:
        self.undertest = Vocab()
        self.tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_add_words(self):
        self.undertest.add_words(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "vocab_data.txt"))
        self.assertEqual(["house", "dog"], list(self.undertest.vocab.keys()))

    def test_add_word(self):
        self.undertest.add_word("test", cnt=31, vec=[1.42, 1.44, 1.55])
        self.assertEqual(["test"], list(self.undertest.vocab.keys()))
        self.assertTrue("test" in self.undertest)

    def test_count(self):
        self.undertest.add_words(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "vocab_data.txt"))
        self.assertEqual(34444, self.undertest.count("house"))

    def test_save_and_load(self):
        self.undertest.add_words(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "vocab_data.txt"))
        self.undertest.add_word("test", cnt=31, vec=[1.42, 1.44, 1.55])
        vocab_path = f"{self.tmp_dir}/vocab.dat"
        self.undertest.save(vocab_path)
        vocab = Vocab.load(vocab_path)
        self.assertEqual(["house", "dog", "test"], list(vocab.vocab.keys()))


class VocabUnigramTableTests(unittest.TestCase):
    EXAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     "..", "examples", "vocab_data.txt")
    UNIGRAM_TABLE_SIZE = 10_000
    # found that this seed had the closest frequency at the sample size we're at
    RANDOM_SEED = 32
    NUM_SAMPLES = 20 # NOTE: 3, 9, 18, and 27 at a time are regular due to context vector sizes
    NUM_TIMES = 200
    # based on the counts on vocab_data.txt and the ones set in setUpClass
    # plus the power of 3/4
    EXPECTED_FREQUENCIES = {
        0: 0.61078822, 1: 0.3182886,
        2: 0.05260281,
        # NOTE: no 3 since that's got no vectors
        4: 0.01832037}
    TOLERANCE = 0.001

    @classmethod
    def setUpClass(cls):
        cls.vocab = Vocab()
        cls.vocab.add_words(cls.EXAMPLE_DATA_PATH)
        cls.vocab.add_word("test", cnt=1310, vec=[1.42, 1.44, 1.55])
        cls.vocab.add_word("vectorless", cnt=1234, vec=None)
        cls.vocab.add_word("withvector", cnt=321, vec=[1.3, 1.2, 0.8])
        cls.vocab.make_unigram_table(table_size=cls.UNIGRAM_TABLE_SIZE)

    def setUp(self):
        np.random.seed(self.RANDOM_SEED)

    @classmethod
    def _get_freqs(cls) -> dict[int, float]:
        c = Counter()
        for _ in range(cls.NUM_TIMES):
            got = cls.vocab.get_negative_samples(cls.NUM_SAMPLES)
            c += Counter(got)
        total = sum(c.values())
        got_freqs = {index: val/total for index, val in c.items()}
        return got_freqs

    @classmethod
    def _get_abs_max_diff(cls, dict1: dict[int, float],
                          dict2: dict[int, float]):
        assert dict1.keys() == dict2.keys()
        vals1, vals2 = [], []
        for index in dict1:
            vals1.append(dict1[index])
            vals2.append(dict2[index])
        return np.max(np.abs(np.array(vals1) - np.array(vals2)))

    def assert_accurate_enough(self, got_freqs: dict[int, float]):
        self.assertEqual(got_freqs.keys(), self.EXPECTED_FREQUENCIES.keys())
        self.assertTrue(
            self._get_abs_max_diff(self.EXPECTED_FREQUENCIES, got_freqs) < self.TOLERANCE)

    def test_does_not_include_vectorless_indices(self, num_samples: int = 100):
        inds = self.vocab.get_negative_samples(num_samples)
        for index in inds:
            with self.subTest(f"Index: {index}"):
                # in the right list
                self.assertIn(index, self.vocab.vec_index2word)
                word = self.vocab.vec_index2word[index]
                info = self.vocab.vocab[word]
                # the info has vector
                self.assertIn("vec", info)
                # the vector is an array or a list
                self.assertIsInstance(self.vocab.vec(word), (np.ndarray, list),)

    def test_negative_sampling(self):
        got_freqs = self._get_freqs()
        self.assert_accurate_enough(got_freqs)


if __name__ == '__main__':
    unittest.main()
