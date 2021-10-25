import os
import shutil
import unittest
import tempfile
from medcat.config import Config
from medcat.cdb_maker import CDBMaker


class CDBTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config = Config()
        config.general["spacy_model"] = "en_core_web_md"
        cls.cdb_maker = CDBMaker(config)

    def setUp(self) -> None:
        cdb_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "cdb.csv")
        cdb_2_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "cdb_2.csv")
        self.tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.undertest = CDBTests.cdb_maker.prepare_csvs([cdb_csv, cdb_2_csv], full_build=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_name2cuis(self):
        self.assertEqual({
            'second~csv': ['C0000239'],
            'virus': ['C0000039', 'C0000139'],
            'virus~k': ['C0000039', 'C0000139'],
            'virus~m': ['C0000039', 'C0000139'],
            'virus~z': ['C0000039', 'C0000139']
        }, self.undertest.name2cuis)

    def test_cui2names(self):
        self.assertEqual({
            'C0000039': {'virus~z', 'virus~k', 'virus~m', 'virus'},
            'C0000139': {'virus~z', 'virus', 'virus~m', 'virus~k'},
            'C0000239': {'second~csv'}
        }, self.undertest.cui2names)

    def test_cui2preferred_name(self):
        self.assertEqual({'C0000039': 'Virus', 'C0000139': 'Virus Z'}, self.undertest.cui2preferred_name)

    def test_cui2type_ids(self):
        self.assertEqual({'C0000039': {'T109', 'T234', 'T123'}, 'C0000139': set(), 'C0000239': set()}, self.undertest.cui2type_ids)

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile() as f:
            self.undertest.save(f.name)
            self.undertest.load(f.name)


if __name__ == '__main__':
    unittest.main()
