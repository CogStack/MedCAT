import os
import shutil
import unittest
import tempfile
import asyncio
import numpy as np
from medcat.config import Config
from medcat.cdb_maker import CDBMaker
from medcat.cdb import CDB


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

    def test_save_async_and_load(self):
        with tempfile.NamedTemporaryFile() as f:
            asyncio.run(self.undertest.save_async(f.name))
            self.undertest.load(f.name)

    def test_empty_count_train(self):
        copied = dict(self.undertest.cui2count_train)
        self.undertest.cui2count_train = {}
        stats = self.undertest.make_stats()
        self.assertFalse(np.isnan(stats["Average training examples per concept"]))
        self.undertest.cui2count_train = copied
    
    def test_remove_cui(self):
        self.undertest.remove_cui('C0000039')
        assert 'C0000039' not in self.undertest.cui2names
        assert 'C0000039' not in self.undertest.cui2snames
        assert 'C0000039' not in self.undertest.cui2count_train
        assert 'C0000039' not in self.undertest.cui2type_ids
        assert 'C0000039' not in self.undertest.cui2preferred_name
        assert 'C0000039' not in self.undertest.name2cuis['virus~z']
        assert 'C0000039' not in self.undertest.name2cuis2status['virus~z']

    def test_cui2snames_population(self):
        self.undertest.cui2snames.clear()
        self.undertest.populate_cui2snames()
        for cui in self.undertest.cui2names:
            with self.subTest(cui):
                self.assertIn(cui, self.undertest.cui2snames)


    def test_merge_cdb(self):
        # generating CDBs
        config = Config()
        maker = CDBMaker(config) 
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_creator", "umls_sample.csv")
        cdb1 = maker.prepare_csvs(csv_paths=[path])
        cdb2 = cdb1.copy()

        # generating vectors and setting up
        zeroes = np.zeros(shape=(1,300))
        ones = np.ones(shape=(1,300))
        for i, cui in enumerate(cdb1.cui2names):
            cdb1.cui2context_vectors[cui] = {"short" : zeroes}
            cdb2.cui2context_vectors[cui] = {"short" : ones}
            cdb1.cui2count_train[cui] = 1
            cdb2.cui2count_train[cui] = i
        test_add = {"test": {'tokens': "test_token", 'snames': "test_sname", 'raw_name': "test_raw_name", "is_upper" : "P"}}
        cdb1.add_names("C0006826", test_add)

        # merging
        cdb = CDB.merge_cdb(cdb1=cdb1, cdb2=cdb2)
        # tests
        


if __name__ == '__main__':
    unittest.main()
