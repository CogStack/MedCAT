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


class OldFormatJsonTests(unittest.TestCase):

    def assert_knows_old_format(self, file_path: str):
        with open(file_path) as f:
            d = json.load(f)
        self.assertTrue(config_utils.is_old_type_config_dict(d))


class OldConfigLoadTests(OldFormatJsonTests):
    JSON_PICKLE_FILE_PATH = os.path.join(
        os.path.dirname(__file__), "..", "resources", "jsonpickle_config.json"
    )
    EXPECTED_VERSION_HISTORY = ['0c0de303b6dc0020',]

    def test_knows_is_old_format(self):
        self.assert_knows_old_format(self.JSON_PICKLE_FILE_PATH)

    def test_loads_old_style_correctly(self):
        cnf: main_config.Config = main_config.Config.load(self.JSON_PICKLE_FILE_PATH)
        self.assertEqual(cnf.version.history, self.EXPECTED_VERSION_HISTORY)


class MetaCATConfigTests(OldFormatJsonTests):
    META_CAT_OLD_PATH = os.path.join(
        os.path.dirname(__file__), "..", "resources", "jsonpickle_meta_cat_config.json"
    )
    EXPECTED_TARGET = -100
    TARGET_CLASS = config_meta_cat.ConfigMetaCAT

    @classmethod
    def get_target(cls, cnf):
        return cnf.general.seed

    def test_knows_is_old_format(self):
        self.assert_knows_old_format(self.META_CAT_OLD_PATH)

    def test_can_load_old_format_correctly(self):
        cnf = self.TARGET_CLASS.load(self.META_CAT_OLD_PATH)
        self.assertIsInstance(cnf, self.TARGET_CLASS)
        self.assertEqual(self.get_target(cnf), self.EXPECTED_TARGET)


class TNERCATConfigTests(MetaCATConfigTests):
    META_CAT_OLD_PATH = os.path.join(
        os.path.dirname(__file__), "..", "resources", "jsonpickle_tner_config.json"
    )
    EXPECTED_TARGET = -100
    TARGET_CLASS = config_transformers_ner.ConfigTransformersNER

    @classmethod
    def get_target(cls, cnf):
        return cnf.general.pipe_batch_size_in_chars


class RelCATConfigTests(MetaCATConfigTests):
    META_CAT_OLD_PATH = os.path.join(
        os.path.dirname(__file__), "..", "resources", "jsonpickle_rel_cat_config.json"
    )
    EXPECTED_TARGET = 100_000
    TARGET_CLASS = config_rel_cat.ConfigRelCAT

    @classmethod
    def get_target(cls, cnf):
        return cnf.train.lr
