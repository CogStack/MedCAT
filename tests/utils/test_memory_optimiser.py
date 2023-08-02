from medcat.utils import memory_optimiser

import unittest
import tempfile
import os
import shutil
import json
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.utils.saving import coding


class DelegatingDictTests(unittest.TestCase):
    _dict = {'c1': [None, 0], 'c2': [1, None]}

    def setUp(self) -> None:
        # deep copy so that the origianl remains unchangeds
        _dict = dict((k, v.copy()
                      ) for k, v in self._dict.items())
        self.del_dict1 = memory_optimiser.DelegatingDict(_dict, 0, 2)
        self.del_dict2 = memory_optimiser.DelegatingDict(_dict, 1, 2)
        self.delegators = [self.del_dict1, self.del_dict2]
        self.names = ['delegator 1', 'delegator 2']
        self.expected_lens = [len(
            [v[nr] for v in _dict.values() if v[nr] is not None]
        ) for nr in range(len(_dict[list(_dict.keys())[0]]))]

    def test_removal(self, key='c2'):
        self.assertIn(key, self.del_dict1)
        del self.del_dict1[key]
        self.assertNotIn(key, self.del_dict1)

    def test_pop_no_def_existing(self, key='c2'):
        self.assertIn(key, self.del_dict1)
        val = self.del_dict1.pop(key)
        self.assertNotIn(key, self.del_dict1)
        self.assertIs(val, self._dict[key][0])

    def test_pop_def_non_existing(self, key='c1', def_val='DEF VAL'):
        self.assertNotIn(key, self.del_dict1)
        val = self.del_dict1.pop(key, def_val)
        self.assertNotIn(key, self.del_dict1)
        self.assertIs(val, def_val)

    def test_adding_exiting_key_nonexist_value(self, key: str = 'c1'):
        self.assertNotIn(key, self.del_dict1)
        self.del_dict1[key] = 'value'
        self.assertIn(key, self.del_dict1)

    def test_adding_nonexiting_key(self, key: str = 'nek1'):
        self.assertNotIn(key, self.del_dict1)
        self.del_dict1[key] = 'value-NEW'
        self.assertIn(key, self.del_dict1)

    def test_adding_nonexiting_key_not_affect_other(self, key: str = 'nek2'):
        self.assertNotIn(key, self.del_dict2)
        self.del_dict1[key] = 'value-NEW-2'
        self.assertNotIn(key, self.del_dict2)

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


class DelegatingDictJsonTests(unittest.TestCase):
    _dict = {'c5': [None, 10], 'c6': [11, None]}

    def setUp(self) -> None:
        self.del_dict1 = memory_optimiser.DelegatingDict(self._dict, 0, 2)
        self.del_dict2 = memory_optimiser.DelegatingDict(self._dict, 1, 2)
        self.delegators = [self.del_dict1, self.del_dict2]
        self.master_dict = {'one2many': self._dict,
                            'part1': self.del_dict1,
                            'part2': self.del_dict2}

    def serialise_master(self) -> str:
        return json.dumps(self.master_dict,
                          cls=coding.CustomDelegatingEncoder.def_inst)

    def deserialise(self, s: str, one2many_name='one2many') -> dict:
        d = json.loads(s, object_hook=coding.default_hook)
        one2many = d[one2many_name]
        for key, value in d.items():
            if key == one2many_name:
                continue
            if value.delegate is None:
                value.delegate = one2many
        return d

    def test_dict_of_delegation_serialises(self):
        s = self.serialise_master()
        self.assertIsInstance(s, str)

    def test_dod_ser_has_keys(self):
        s = self.serialise_master()
        for key in self.master_dict:
            with self.subTest(key):
                self.assertIn(key, s)

    def test_dod_ser_one2many_has_sub_keys(self):
        s = self.serialise_master()
        for key in self.master_dict['one2many']:
            with self.subTest(key):
                self.assertIn(key, s)

    def test_round_trip(self):
        s = self.serialise_master()
        d = self.deserialise(s)
        self.assertIsInstance(d, dict)

    def test_round_trip_equal(self):
        s = self.serialise_master()
        d = self.deserialise(s)
        self.assertEqual(d, self.master_dict)


class UnOptimisingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cdb = CDB.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "examples", "cdb.dat"))

    def test_unoptimised_cdb_does_not_have_cui2many(self):
        self.assertFalse(hasattr(self.cdb, 'cui2many'))

    def test_unoptmised_cdb_does_not_have_delegating_dicts(self):
        for key, val in self.cdb.__dict__.items():
            with self.subTest(key):
                self.assertNotIsInstance(val, memory_optimiser.DelegatingDict)

    def test_unoptimised_knows_has_no_optimsied_parts(self):
        self.assertFalse(self.cdb._memory_optimised_parts,
                         "Should have empty optimised partss")

    def test_simply_loaded_model_not_dirty(self):
        self.assertFalse(self.cdb.is_dirty)


class MemoryOptimisingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cdb = CDB.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "examples", "cdb.dat"))
        memory_optimiser.perform_optimisation(cls.cdb, optimise_snames=True)

    def test_is_dirty(self):
        self.assertTrue(self.cdb.is_dirty,
                        "Should be dirty after optimisation")

    def test_knows_optimised(self):
        self.assertTrue(self.cdb._memory_optimised_parts,
                        "Should have non-empty `_memory_optimised_parts`")

    def test_knows_correct_parts_optimsed(self, should_be=['CUIS', 'snames']):
        for name in should_be:
            with self.subTest(name):
                self.assertIn(name, self.cdb._memory_optimised_parts)

    def test_knows_incorrect_parts_NOT_optimised(self, should_not_be=['NAMES']):
        for name in should_not_be:
            with self.subTest(name):
                self.assertNotIn(name, self.cdb._memory_optimised_parts)

    def test_cdb_has_one2many(self, one2many_name='cui2many'):
        self.assertTrue(hasattr(self.cdb, one2many_name))
        one2many = getattr(self.cdb, one2many_name)
        self.assertIsInstance(one2many, dict)

    def test_cdb_has_delegating_dicts(self):
        for dict_name in memory_optimiser.CUI_DICT_NAMES_TO_COMBINE:
            with self.subTest(dict_name):
                d = getattr(self.cdb, dict_name)
                self.assertIsInstance(d, memory_optimiser.DelegatingDict)

    def test_has_delegating_set(self):
        self.assertIsInstance(
            self.cdb.snames, memory_optimiser.DelegatingValueSet)

    def test_delegating_set_has_values(self):
        for values in self.cdb.cui2snames.values():
            for val in values:
                with self.subTest(f'Checking {val}'):
                    self.assertIn(val, self.cdb.snames)


class MemoryUnoptimisingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cdb = CDB.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "examples", "cdb.dat"))

    def test_optimisation_round_trip_cuis(self):
        cui_dicts_before = [getattr(self.cdb, dict_name)
                            for dict_name in memory_optimiser.CUI_DICT_NAMES_TO_COMBINE]
        memory_optimiser.perform_optimisation(self.cdb)
        memory_optimiser.unoptimise_cdb(self.cdb)
        cui_dicts_after = [getattr(self.cdb, dict_name)
                           for dict_name in memory_optimiser.CUI_DICT_NAMES_TO_COMBINE]
        for before, after, name in zip(cui_dicts_before,
                                       cui_dicts_after,
                                       memory_optimiser.CUI_DICT_NAMES_TO_COMBINE):
            with self.subTest(f'{name}'):
                self.assertIsInstance(before, dict)
                self.assertIsInstance(after, dict)
                self.assertEquals(len(before), len(after))
                self.assertEquals(before, after)

    def test_optimisation_round_trip_snames(self):
        snames_before = self.cdb.snames
        memory_optimiser.perform_optimisation(self.cdb)
        memory_optimiser.unoptimise_cdb(self.cdb)
        snames_after = self.cdb.snames
        self.assertIsInstance(snames_before, set)
        self.assertIsInstance(snames_after, set)
        self.assertEquals(len(snames_before), len(snames_after))
        self.assertEquals(snames_before, snames_after)

    def test_optimisation_round_trip_dirty(self):
        memory_optimiser.perform_optimisation(self.cdb)
        memory_optimiser.unoptimise_cdb(self.cdb)
        self.assertTrue(self.cdb.is_dirty)

    def test_optimisation_round_trip_no_optimised_parts(self):
        memory_optimiser.perform_optimisation(self.cdb)
        memory_optimiser.unoptimise_cdb(self.cdb)
        self.assertFalse(self.cdb._memory_optimised_parts,
                         "Should have no optimised parts")


class OperationalTests(unittest.TestCase):
    temp_folder = tempfile.TemporaryDirectory()
    temp_cdb_path = os.path.join(temp_folder.name, 'cat.cdb')
    json_path = temp_cdb_path.rsplit(os.path.sep, 1)[0]
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
        memory_optimiser.perform_optimisation(cls.cdb, optimise_snames=True)
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

    def test_optimised_cdb_has_cui2many(self):
        self.assertTrue(hasattr(self.cdb, 'cui2many'))

    def test_can_be_saved_as_json(self):
        self.cdb.save(self.temp_cdb_path, json_path=self.json_path)

    def test_can_be_loaded_as_json(self):
        self.test_can_be_saved_as_json()
        cdb = CDB.load(self.temp_cdb_path, self.json_path)
        self.assertEqual(self.cdb.cui2many, cdb.cui2many)
        for del_name in memory_optimiser.CUI_DICT_NAMES_TO_COMBINE:
            d = getattr(cdb, del_name)
            with self.subTest(del_name):
                self.assertIsInstance(d, memory_optimiser.DelegatingDict)
                self.assertIs(cdb.cui2many, d.delegate)


class DelegatingValueSetTests(unittest.TestCase):

    def setUp(self) -> None:
        self.delegate = {'a': set('abcd'),
                         'b': set('efghij'),
                         'c': set('lm'),  # skip k
                         'd': set('qrst'),  # skip a bunch
                         }
        self.original = set([v for s in self.delegate for v in s])

    def test_DelegatingValueSet_constructs(self):
        dvs = memory_optimiser.DelegatingValueSet(self.delegate)
        self.assertIsInstance(dvs, memory_optimiser.DelegatingValueSet)

    def test_DelegatingValueSet_contains_values(self):
        dvs = memory_optimiser.DelegatingValueSet(self.delegate)
        for v in self.original:
            with self.subTest(f'Check: {v}'):
                self.assertIn(v, dvs)

    def test_DelegatingValueSet_contains_incorrect_values(self,
                                                          to_check=set('kopuvwxyz')):
        dvs = memory_optimiser.DelegatingValueSet(self.delegate)
        for v in to_check:
            with self.subTest(f'Check: {v}'):
                self.assertNotIn(v, dvs)
