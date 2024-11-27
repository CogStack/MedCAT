from medcat.vocab import Vocab
from medcat.utils import vocab_utils
from medcat.cdb import CDB

import unittest
import numpy as np
import os
import random


WORDS = [
    ("word1", 12, np.array([0, 1, 2, 1, 1, 0])),
    ("word2", 21, np.array([2, -1, 0, 1, -1, -1])),
    ("word3", 32, np.array([2, -1, 0, 0, 0, 1])),
    ("word4", 42, np.array([-1, 0, -1, -1, 0, 2])),
    ("word5", 24, np.array([0, 3, -2, 5, -1, 3])),
    ("word6", 46, np.array([3, -5, 10, 1, 10, -2])),
    ("word7", 31, np.array([-2, 4, -1, -2, 1, 2])),
    ("word8", 28, np.array([-3, 3, -2, 4, 9, 2])),
    ("word9", 19, np.array([-4, 2, -3, -6, 3, 2])),
    ("word10", 1, np.array([4, 1, -4, 0, 5, 2])),
]


class TestWithTransformationMatrixBase(unittest.TestCase):
    ORIG_SIZE = len(WORDS[0][-1])
    TARGET_SIZE = 3

    @classmethod
    def setUpClass(cls):
        cls.vocab = Vocab()
        for word, cnt, vec in WORDS:
            cls.vocab.add_word(word, cnt, vec)
        cls.TM = vocab_utils.calc_matrix(cls.vocab, cls.TARGET_SIZE)


class TransformationMatrixTests(TestWithTransformationMatrixBase):

    def test_transformation_matrix_correct_size(self):
        self.assertEqual(self.TM.shape, (self.TARGET_SIZE, self.ORIG_SIZE))

    def test_transformation_matrix_reasonable(self):
        self.assertFalse(np.any(self.TM != self.TM), "Shouldn't have NaNs")
        self.assertFalse(np.any(self.TM - 100 == self.TM), "Shouldn't have infinity")


class TestWithTMAndCDBBase(TestWithTransformationMatrixBase):
    CDB_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                            "examples", "cdb.dat")
    UNIGRAM_TABLE_SIZE = 100

    @classmethod
    def add_fake_context_vectors(cls, words: int = 4):
        # NOTE: in original size!
        cui2cv = cls.cdb.cui2context_vectors
        for cui in cls.cdb.cui2names:
            cui_cv = {}
            for cv_type in cls.cdb.config.linking.context_vector_sizes:
                cv = 0
                for _ in range(words):
                    # get the original vector
                    cv += random.choice(WORDS)[2]
                cui_cv[cv_type] = cv
            cui2cv[cui] = cui_cv

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cdb = CDB.load(cls.CDB_PATH)
        cls.add_fake_context_vectors()


class VocabTransformationTests(TestWithTMAndCDBBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.do_conversion()

    @classmethod
    def do_conversion(cls):
        vocab_utils.convert_vocab(cls.vocab, cls.TM,
                                  unigram_table_size=cls.UNIGRAM_TABLE_SIZE)
        vocab_utils.convert_context_vectors(cls.cdb, cls.TM)

    def test_can_transform_vocab(self):
        for w in self.vocab.vocab:
            with self.subTest(w):
                vec = self.vocab.vec(w)
                self.assertEqual(len(vec), self.TARGET_SIZE)

    def test_can_transform_cdb(self):
        for cui, cv in self.cdb.cui2context_vectors.items():
            for cvt, vec in cv.items():
                with self.subTest(f"{cui}-{cvt}"):
                    self.assertEqual(len(vec), self.TARGET_SIZE)


class OverallTransformationTests(VocabTransformationTests):

    @classmethod
    def do_conversion(cls):
        vocab_utils.convert_vocab_vector_size(cls.cdb, cls.vocab,
                                              cls.TARGET_SIZE)
