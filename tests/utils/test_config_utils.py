from medcat.config import Config
from medcat.utils.saving.coding import default_hook, CustomDelegatingEncoder
from medcat.utils import config_utils
from medcat import config as main_config
from medcat import config_meta_cat
from medcat import config_transformers_ner
from medcat import config_rel_cat
import json
import os

import unittest

OLD_STYLE_DICT = {'py/object': 'medcat.config.VersionInfo',
                  'py/state': {
                      '__dict__': {
                          'history': ['0c0de303b6dc0020',],
                          'meta_cats': [],
                          'cdb_info': {
                              'Number of concepts': 785910,
                              'Number of names': 2480049,
                              'Number of concepts that received training': 378746,
                              'Number of seen training examples in total': 1863973060,
                              'Average training examples per concept': {
                                  'py/reduce': [{'py/function': 'numpy.core.multiarray.scalar'},]
                                  }
                              },
                          'performance': {'ner': {}, 'meta': {}},
                          'description': 'No description',
                          'id': 'ff4f4e00bc97de58',
                          'last_modified': '26 April 2024',
                          'location': None,
                          'ontology': ['ONTOLOGY1'],
                          'medcat_version': '1.10.2'
                          },
                      '__fields_set__': {
                          'py/set': ['id', 'ontology', 'description', 'history',
                                     'location', 'medcat_version', 'last_modified',
                                     'meta_cats', 'cdb_info', 'performance']
                                     },
                      '__private_attribute_values__': {}
                    }
                 }


NEW_STYLE_DICT = json.loads(json.dumps(Config().asdict(), cls=CustomDelegatingEncoder.def_inst),
                            object_hook=default_hook)


class ConfigUtilsTests(unittest.TestCase):

    def test_identifies_old_style_dict(self):
        self.assertTrue(config_utils.is_old_type_config_dict(OLD_STYLE_DICT))

    def test_identifies_new_style_dict(self):
        self.assertFalse(config_utils.is_old_type_config_dict(NEW_STYLE_DICT))


class ConfigRemapperGeneralTests(unittest.TestCase):
    ORIG_DICT = {'a': {'a1': 1, 'a2': 2}, 'b': {'b1': 3, 'b2': 4, 'b4': 5}}
    EXAMPLE_MAPPINGS = {'c': {'a1_from_a': 'a.a1', 'b2_from_b': 'b.b2', 'b4_from_b': 'b.b4'}}
    EXPECTED_OUT = {'a': {'a2': 2}, 'b': {'b1': 3}, 'c': {'a1_from_a': 1, 'b2_from_b': 4, 'b4_from_b': 5}}
    EXPECTED_NEW = {'c': {'a1_from_a': 1, 'b2_from_b': 4, 'b4_from_b': 5}}

    def test_remapping_works(self):
        got = config_utils.remap_nested_dict(self.ORIG_DICT, self.EXAMPLE_MAPPINGS)
        self.assertEqual(got, self.EXPECTED_OUT)

    def test_remapping_into_new_works(self):
        got = config_utils.remap_nested_dict(self.ORIG_DICT, self.EXAMPLE_MAPPINGS, in_place=False)
        self.assertEqual(got, self.EXPECTED_NEW)


class ConfigRemapWithConfigTests(unittest.TestCase):
    CONFIG_JSON_PATH = os.path.join(
        os.path.dirname(__file__), "..", "resources", "pre_change_config.json"
    )
    EXPECTED_SPACY_MODEL = "no_such_spacy_model"

    @classmethod
    def setUpClass(cls) -> None:
        cls.config: main_config.Config = main_config.Config.load(cls.CONFIG_JSON_PATH)

    def test_gets_correct_spacy(self):
        self.assertEqual(self.config.pre_load.spacy_model, self.EXPECTED_SPACY_MODEL)

    def test_does_not_have_spacy_in_old_path(self):
        self.assertFalse(hasattr(self.config.general, "spacy_model"))


class ConfigRemapWithMetaCATConfigTests(unittest.TestCase):
    CONFIG_JSON_PATH = os.path.join(
        os.path.dirname(__file__), "..", "resources", "pre_change_meta_cat_config.json"
    )
    EXPECTED_SEED = -130

    @classmethod
    def setUpClass(cls) -> None:
        cls.config: config_meta_cat.ConfigMetaCAT = config_meta_cat.ConfigMetaCAT.load(cls.CONFIG_JSON_PATH)

    def test_gets_correct_spacy(self):
        self.assertEqual(self.config.pre_load.seed, self.EXPECTED_SEED)

    def test_does_not_have_spacy_in_old_path(self):
        self.assertFalse(hasattr(self.config.general, "seed"))


class ConfigRemapWithTNERConfigTests(unittest.TestCase):
    CONFIG_JSON_PATH = os.path.join(
        os.path.dirname(__file__), "..", "resources", "pre_change_rel_cat_config.json"
    )
    EXPECTED_SEED = -113

    @classmethod
    def setUpClass(cls) -> None:
        cls.config: config_rel_cat.ConfigRelCAT = config_rel_cat.ConfigRelCAT.load(cls.CONFIG_JSON_PATH)

    def test_gets_correct_spacy(self):
        self.assertEqual(self.config.pre_load.seed, self.EXPECTED_SEED)

    def test_does_not_have_spacy_in_old_path(self):
        self.assertFalse(hasattr(self.config.general, "seed"))
