import unittest
import os

from medcat.utils.versioning import get_version_from_modelcard, get_semantic_version
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.vocab import Vocab

from .regression.test_metadata import MODEL_CARD_EXAMPLE, EXAMPLE_VERSION


class VersionGettingFromModelCardTests(unittest.TestCase):
    FAKE_MODEL_CARD1 = {"Something": "value"}
    FAKE_MODEL_CARD2 = {"MedCAT Version": "not semantic"}
    FAKE_MODEL_CARD3 = {"MedCAT Version": "almost.semantic"}
    FAKE_MODEL_CARD4 = {"MedCAT Version": "closest.to.semantic"}
    WRONG_VERSION_FAKE_MODELS = [FAKE_MODEL_CARD2,
                                 FAKE_MODEL_CARD3, FAKE_MODEL_CARD4]

    def test_gets_correct_version(self):
        maj, minor, patch = get_version_from_modelcard(MODEL_CARD_EXAMPLE)
        self.assertEqual(EXAMPLE_VERSION, (maj, minor, patch))

    def test_fails_upon_model_card_with_no_version_defined(self):
        with self.assertRaises(KeyError):
            get_version_from_modelcard(self.FAKE_MODEL_CARD1)

    def test_fails_upon_model_card_with_incorrect_version(self):
        for fake_model_card in self.WRONG_VERSION_FAKE_MODELS:
            with self.assertRaises(ValueError):
                get_version_from_modelcard(fake_model_card)


class VersionGettingFromCATTests(unittest.TestCase):

    def setUp(self) -> None:
        self.cdb = CDB.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "examples", "cdb.dat"))
        self.vocab = Vocab.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "examples", "vocab.dat"))
        self.cdb.config.general.spacy_model = "en_core_web_md"
        self.cdb.config.version.medcat_version = ".".join(str(v) for v in EXAMPLE_VERSION)
        self.cdb.config.ner.min_name_len = 2
        self.cdb.config.ner.upper_case_limit_len = 3
        self.cdb.config.general.spell_check = True
        self.cdb.config.linking.train_count_threshold = 10
        self.cdb.config.linking.similarity_threshold = 0.3
        self.cdb.config.linking.train = True
        self.cdb.config.linking.disamb_length_limit = 5
        self.cdb.config.general.full_unlink = True
        self.meta_cat_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "tmp")
        self.undertest = CAT(
            cdb=self.cdb, config=self.cdb.config, vocab=self.vocab, meta_cats=[])

    def test_gets_correct_version(self):
        version = get_semantic_version(self.undertest)
        self.assertEqual(EXAMPLE_VERSION, version)
