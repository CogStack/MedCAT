import os
import shutil
import unittest
from medcat.vocab import Vocab


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


if __name__ == '__main__':
    unittest.main()
