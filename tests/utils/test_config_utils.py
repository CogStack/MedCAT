from medcat.config import Config
from medcat.utils.saving.coding import default_hook, CustomDelegatingEncoder
from medcat.utils import config_utils
import json

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
