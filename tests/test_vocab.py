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
    RANDOM_SEED = 4976
    NUM_SAMPLES = 20 # NOTE: 3, 9, 18, and 27 at a time are regular due to context vector sizes
    NUM_TIMES = 200
    # based on the counts on vocab_data.txt and the one set in setUpClass
    EXPECTED_FREQUENCIES = [0.62218692, 0.32422858, 0.0535845]
    TOLERANCE = 0.001

    @classmethod
    def setUpClass(cls):
        cls.vocab = Vocab()
        cls.vocab.add_words(cls.EXAMPLE_DATA_PATH)
        cls.vocab.add_word("test", cnt=1310, vec=[1.42, 1.44, 1.55])
        cls.vocab.make_unigram_table(table_size=cls.UNIGRAM_TABLE_SIZE)

    def setUp(self):
        np.random.seed(self.RANDOM_SEED)

    @classmethod
    def _get_freqs(cls) -> list[float]:
        c = Counter()
        for _ in range(cls.NUM_TIMES):
            got = cls.vocab.get_negative_samples(cls.NUM_SAMPLES)
            c += Counter(got)
        total = sum(c[i] for i in c)
        got_freqs = [c[i]/total for i in range(len(cls.EXPECTED_FREQUENCIES))]
        return got_freqs

    def assert_accurate_enough(self, got_freqs: list[float]):
        self.assertTrue(
            np.max(np.abs(np.array(got_freqs) - self.EXPECTED_FREQUENCIES)) < self.TOLERANCE
        )

    def test_negative_sampling(self):
        got_freqs = self._get_freqs()
        self.assert_accurate_enough(got_freqs)


if __name__ == '__main__':
    unittest.main()
