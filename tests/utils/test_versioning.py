import unittest
import os

from medcat.utils.versioning import get_version_from_modelcard, get_semantic_version_from_model
from medcat.utils.versioning import get_version_from_cdb_dump, get_version_from_modelpack_zip
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.vocab import Vocab

from .regression.test_metadata import MODEL_CARD_EXAMPLE, EXAMPLE_VERSION


CORRECT_SEMANTIC_VERSIONS = [("1.0.1-alpha-1", (1, 0, 1)), ("0.0.1-alpha-1", (0, 0, 1)),
                             ("1.0.0-alpha.1", (1, 0, 0)
                              ), ("1.0.0-0.3.7", (1, 0, 0)),
                             ("1.0.0-x.7.z.92", (1, 0, 0)
                              ), ("1.0.0-x-y-z.--", (1, 0, 0)),
                             ("1.0.0-alpha+001", (1, 0, 0)
                              ), ("1.0.0+20130313144700", (1, 0, 0)),
                             ("1.0.0-beta+exp.sha.5114f85", (1, 0, 0)),
                             ("1.0.0+21AF26D3----117B344092BD", (1, 0, 0))]
INCORRECT_SEMANTIC_VERSIONS = ["01.0.0", "0.01.0", "0.0.01", "0.0.0\nSOMETHING",
                               "1.0.space", "1.0.0- space"]


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
        cntr = 0
        for fake_model_card in self.WRONG_VERSION_FAKE_MODELS:
            with self.assertRaises(ValueError):
                get_version_from_modelcard(fake_model_card)
            cntr += 1
        self.assertEqual(cntr, len(self.WRONG_VERSION_FAKE_MODELS))

    def test_fails_upon_wrong_version(self):
        cntr = 0
        for wrong_version in INCORRECT_SEMANTIC_VERSIONS:
            d = {"MedCAT Version": wrong_version}
            with self.subTest(f"With version: {wrong_version}"):
                with self.assertRaises(ValueError):
                    get_version_from_modelcard(d)
                cntr += 1
        self.assertEqual(cntr, len(INCORRECT_SEMANTIC_VERSIONS))

    def test_gets_version_from_correct_versions(self):
        cntr = 0
        for version, expected in CORRECT_SEMANTIC_VERSIONS:
            d = {"MedCAT Version": version}
            with self.subTest(f"With version: {version}"):
                got_version = get_version_from_modelcard(d)
                self.assertEqual(got_version, expected)
                cntr += 1
        self.assertEqual(cntr, len(CORRECT_SEMANTIC_VERSIONS))


NEW_CDB_NAME = "cdb_new.dat"
CDB_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "..", "..", "examples", NEW_CDB_NAME)
EXPECTED_CDB_VERSION = (1, 0, 0)


class VersionGettingFromCATTests(unittest.TestCase):

    def setUp(self) -> None:
        self.cdb = CDB.load(CDB_PATH)
        self.vocab = Vocab.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "examples", "vocab.dat"))
        self.cdb.config.general.spacy_model = "en_core_web_md"
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
        version = get_semantic_version_from_model(self.undertest)
        self.assertEqual(EXPECTED_CDB_VERSION, version)


class VersionGetterFromCDBTests(unittest.TestCase):

    def test_gets_version_from_cdb(self):
        version = get_version_from_cdb_dump(CDB_PATH)
        self.assertEqual(EXPECTED_CDB_VERSION, version)


class VersionGettFromModelPackTests(unittest.TestCase):

    def test_gets_version_from_model_pack(self):
        # not strictly speaking a ZIP, but should work currently
        # since the folder exists
        model_pack_zip = os.path.dirname(CDB_PATH)
        version = get_version_from_modelpack_zip(model_pack_zip, cdb_file_name=NEW_CDB_NAME)
        self.assertEqual(EXPECTED_CDB_VERSION, version)
