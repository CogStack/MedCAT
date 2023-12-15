import unittest
import numpy as np
from tests.helper import ForCDBMerging
from medcat.utils.cdb_utils import merge_cdb


class CDBMergeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        to_merge = ForCDBMerging()
        cls.cdb1 = to_merge.cdb1
        cls.cdb2 = to_merge.cdb2
        cls.merged_cdb = merge_cdb(cdb1=cls.cdb1, cdb2=cls.cdb2)
        cls.overwrite_cdb = merge_cdb(cdb1=cls.cdb1, cdb2=cls.cdb2, overwrite_training=2, full_build=True)
        cls.zeroes = np.zeros(shape=(1,300))
        cls.ones = np.ones(shape=(1,300))

    def test_merge_inserts(self):
        self.assertIn("test", self.merged_cdb.cui2names["C0006826"])
        self.assertIn("test_name", self.merged_cdb.cui2snames["C0006826"])
        self.assertEqual("Cancer", self.merged_cdb.cui2preferred_name["C0006826"])

    def test_no_full_build(self):
        self.assertEqual(self.merged_cdb.addl_info["cui2ontologies"], dict())
        self.assertEqual(self.merged_cdb.addl_info["cui2ontologies"], dict())

    def test_full_build(self):
        for cui in self.cdb2.cui2names:
            self.assertEqual(self.overwrite_cdb.addl_info["cui2ontologies"][cui], {"test_ontology"})
            self.assertEqual(self.overwrite_cdb.addl_info["cui2description"][cui], "test_description")

    def test_vector_merge(self):
        self.assertTrue(np.array_equal(self.zeroes, self.merged_cdb.cui2context_vectors["UniqueTest"]["short"]))
        for i, cui in enumerate(self.cdb1.cui2names):
            self.assertTrue(np.array_equal(self.merged_cdb.cui2context_vectors[cui]["short"], np.divide(self.ones, i+2)))


    def test_overwrite_parameter(self):
        for cui in self.cdb2.cui2names:
            self.assertTrue(np.array_equal(self.overwrite_cdb.cui2context_vectors[cui]["short"], self.zeroes))
            self.assertEqual(self.overwrite_cdb.addl_info["cui2ontologies"][cui], {"test_ontology"})
            self.assertEqual(self.overwrite_cdb.addl_info["cui2description"][cui], "test_description")
