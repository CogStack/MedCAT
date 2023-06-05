from medcat.utils import memory_optimiser

import unittest
import tempfile
import os
import shutil
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.vocab import Vocab


class DelegatingDictTests(unittest.TestCase):
    _dict = {'c1': [None, 0], 'c2': [1, None]}

    def setUp(self) -> None:
        self.del_dict1 = memory_optimiser.DelegatingDict(self._dict, 0, 2)
        self.del_dict2 = memory_optimiser.DelegatingDict(self._dict, 1, 2)
        self.delegators = [self.del_dict1, self.del_dict2]
        self.names = ['delegator 1', 'delegator 2']
        self.expected_lens = [len(
            [v[nr] for v in self._dict.values() if v[nr] is not None]
        ) for nr in range(len(self._dict[list(self._dict.keys())[0]]))]

    def test_delegating_dict_has_correct_keys(self):
        for delegator, exp_len, name in zip(self.delegators, self.expected_lens, self.names):
            with self.subTest(name):
                self.assertEqual(len(delegator.keys()), exp_len)

    def test_delegating_dict_has_same_number_of_keys_and_values(self):
        for delegator, exp_len, name in zip(self.delegators, self.expected_lens, self.names):
            with self.subTest(name):
                self.assertEqual(len(delegator.keys()), exp_len)
                self.assertEqual(len(delegator.values()), exp_len)

    def test_delegating_dict_has_same_number_of_items_and_iter_values(self):
        for delegator, exp_len, name in zip(self.delegators, self.expected_lens, self.names):
            with self.subTest(name):
                self.assertEqual(len(delegator.items()), exp_len)
                # __iter__ -> list -> len
                self.assertEqual(len(list(delegator)), exp_len)

    def test_delegator_do_not_have_None_values(self):
        for delegator, name in zip(self.delegators, self.names):
            for key, val in delegator.items():
                with self.subTest(f"{name}: {key}"):
                    self.assertIsNotNone(val)

    def test_delegator_keys_in_original(self):
        for delegator, name in zip(self.delegators, self.names):
            for key in delegator.keys():
                with self.subTest(f"{name}: {key}"):
                    self.assertIn(key, self._dict)

    def test_delegator_keys_in_container(self):
        for delegator, name in zip(self.delegators, self.names):
            for key in delegator.keys():
                with self.subTest(f"{name}: {key}"):
                    self.assertIn(key, delegator)

    def test_delegator_get_gets_key(self, def_value='#DEFAULT#'):
        for delegator, name in zip(self.delegators, self.names):
            for key in delegator.keys():
                with self.subTest(f"{name}: {key}"):
                    val = delegator.get(key, def_value)
                    self.assertIsNot(val, def_value)

    def test_delegator_get_defaults_non_existant_key(self, def_value='#DEFAULT#'):
        for delegator, name in zip(self.delegators, self.names):
            for key in self._dict.keys():
                if key in delegator:
                    continue
                with self.subTest(f"{name}: {key}"):
                    val = delegator.get(key, def_value)
                    self.assertIs(val, def_value)


class MemoryOptimisingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cdb = CDB.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "examples", "cdb.dat"))
        memory_optimiser.perform_optimisation(cls.cdb)

    def test_cdb_has_one2many(self, one2many_name='cui2many'):
        self.assertTrue(hasattr(self.cdb, one2many_name))
        one2many = getattr(self.cdb, one2many_name)
        self.assertIsInstance(one2many, dict)

    def test_cdb_has_delegating_dicts(self):
        for dict_name in memory_optimiser.CUI_DICT_NAMES_TO_COMBINE:
            with self.subTest(dict_name):
                d = getattr(self.cdb, dict_name)
                self.assertIsInstance(d, memory_optimiser.DelegatingDict)


class OperationalTests(unittest.TestCase):
    temp_folder = tempfile.TemporaryDirectory()
    temp_cdb_path = os.path.join(temp_folder.name, 'cat.cdb')
    # importing here so it's in the local namespace
    # otherwise, all of its parts would get run again
    from tests.test_cat import CATTests
    test_callable_with_single_text = CATTests.test_callable_with_single_text
    test_callable_with_single_empty_text = CATTests.test_callable_with_single_empty_text
    test_callable_with_single_none_text = CATTests.test_callable_with_single_none_text
    test_get_entities = CATTests.test_get_entities
    test_get_entities_including_text = CATTests.test_get_entities_including_text
    test_get_entities_multi_texts = CATTests.test_get_entities_multi_texts
    test_get_entities_multi_texts_including_text = CATTests.test_get_entities_multi_texts_including_text

    @classmethod
    def setUpClass(cls) -> None:
        cls.cdb = CDB.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "examples", "cdb.dat"))
        memory_optimiser.perform_optimisation(cls.cdb)
        cls.vocab = Vocab.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "examples", "vocab.dat"))
        cls.cdb.config.general.spacy_model = "en_core_web_md"
        cls.cdb.config.ner.min_name_len = 2
        cls.cdb.config.ner.upper_case_limit_len = 3
        cls.cdb.config.general.spell_check = True
        cls.cdb.config.linking.train_count_threshold = 10
        cls.cdb.config.linking.similarity_threshold = 0.3
        cls.cdb.config.linking.train = True
        cls.cdb.config.linking.disamb_length_limit = 5
        cls.cdb.config.general.full_unlink = True
        cls.meta_cat_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "tmp")
        cls.undertest = CAT(cdb=cls.cdb, config=cls.cdb.config,
                            vocab=cls.vocab, meta_cats=[])
        cls._linkng_filters = cls.undertest.config.linking.filters.copy_of()

        # # add tests from CAT tests

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_folder.cleanup()
        cls.undertest.destroy_pipe()
        if os.path.exists(cls.meta_cat_dir):
            shutil.rmtree(cls.meta_cat_dir)

    def tearDown(self) -> None:
        self.cdb.config.annotation_output.include_text_in_output = False
        # need to make sure linking filters are not retained beyond a test scope
        self.undertest.config.linking.filters = self._linkng_filters.copy_of()
