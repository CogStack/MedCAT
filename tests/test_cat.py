import os
import unittest
from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT


class CATTests(unittest.TestCase):

    def setUp(self) -> None:
        self.cdb = CDB.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "cdb.dat"))
        self.vocab = Vocab.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "vocab.dat"))
        self.cdb.config.ner['min_name_len'] = 2
        self.cdb.config.ner['upper_case_limit_len'] = 3
        self.cdb.config.general['spell_check'] = True
        self.cdb.config.linking['train_count_threshold'] = 10
        self.cdb.config.linking['similarity_threshold'] = 0.3
        self.cdb.config.linking['train'] = True
        self.cdb.config.linking['disamb_length_limit'] = 5
        self.cdb.config.general['full_unlink'] = True
        self.undertest = CAT(cdb=self.cdb, config=self.cdb.config, vocab=self.vocab)

    def test_pipeline(self):
        text = "The dog is sitting outside the house."
        doc = self.undertest(text)
        self.assertEqual(text, doc.text)

