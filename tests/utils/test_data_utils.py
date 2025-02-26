import os
import json
from copy import deepcopy

from medcat.utils import data_utils
from medcat.stats.mctexport import count_all_annotations, count_all_docs

from unittest import TestCase


class FakeCDB:

    def __init__(self):
        self.cui2tui = {}

    def get_name(self, cui: str) -> str:
        return cui


class TestTrainSplitTestsBase(TestCase):
    file_name = os.path.join(os.path.dirname(__file__),
        "..", "resources", "medcat_trainer_export.json")
    allowed_doc_ids = {3204, 3205}
    test_size = 0.2
    expect_empty_train_set = False
    expect_empty_test_set = False
    seed = None

    @classmethod
    def setUpClass(cls):
        with open(cls.file_name) as f:
            cls.data = json.load(f)
        cls.undertest = cls.data
        cls.cdb = FakeCDB()

    def setUp(self):
        if self.seed is not None:
            data_utils.set_all_seeds(self.seed)
        (self.train_set, self.test_set,
         self.num_test_anns, 
         self.num_total_anns) = data_utils.make_mc_train_test(
             self.undertest, self.cdb, test_size=self.test_size)


class TestTrainSplitUnfilteredTests(TestTrainSplitTestsBase):

    def test_all_docs_accounted_for(self):
        self.assertEqual(count_all_docs(self.undertest),
                         count_all_docs(self.train_set) + count_all_docs(self.test_set))

    def test_all_anns_accounted_for(self):
        self.assertEqual(count_all_annotations(self.undertest),
                         count_all_annotations(self.train_set) + count_all_annotations(self.test_set))

    def test_total_anns_match(self):
        total = count_all_annotations(self.undertest)
        self.assertEqual(self.num_total_anns, total)
        self.assertEqual(self.num_test_anns + count_all_annotations(self.train_set),
                         total)

    def test_nonempty_train(self):
        if not self.expect_empty_train_set:
            self.assertTrue(self.train_set)
            self.assertTrue(self.num_total_anns - self.num_test_anns)
        self.assertEqual(self.num_total_anns - self.num_test_anns,
                         count_all_annotations(self.train_set))

    def test_nonempty_test(self):
        if not self.expect_empty_test_set:
            self.assertTrue(self.test_set)
            self.assertTrue(self.num_test_anns)
        self.assertEqual(self.num_test_anns,
                         count_all_annotations(self.test_set))


class TestTrainSplitFilteredTestsBase(TestTrainSplitUnfilteredTests):
    expect_empty_test_set = True
    # would work with previous version:
    # seed = 332378110
    # was guaranteed to fail with previous version:
    seed = 73607120

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.filtered = deepcopy(cls.data)
        for proj in cls.filtered['projects']:
            proj['documents'] = [doc for doc in proj['documents']
                                 if doc['id'] in cls.allowed_doc_ids]
        cls.undertest = cls.filtered
