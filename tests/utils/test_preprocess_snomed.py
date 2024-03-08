from typing import Dict
from medcat.utils import preprocess_snomed

import unittest


EXAMPLE_REFSET_DICT: Dict = {
   'SCUI1': [
       {'code': 'TCUI1', 'mapPriority': '1'},
       {'code': 'TCUI2', 'mapPriority': '2'},
       {'code': 'TCUI3', 'mapPriority': '3'},
       ]
}

# in order from highest priority to lowest
EXPECTED_DIRECT_MAPPINGS = {"SCUI1": ['TCUI3', 'TCUI2', 'TCUI1']}

EXAMPLE_REFSET_DICT_WITH_EXTRAS = dict(
    (k, [dict(v, otherKey=f"val-{k}") for v in vals]) for k, vals in EXAMPLE_REFSET_DICT.items())

EXAMPLE_REFSET_DICT_NO_PRIORITY = dict(
    (k, [{ik: iv for ik, iv in v.items() if ik != 'mapPriority'} for v in vals]) for k, vals in EXAMPLE_REFSET_DICT.items()
)

EXAMPLE_REFSET_DICT_NO_CODE = dict(
    (k, [{ik: iv for ik, iv in v.items() if ik != 'code'} for v in vals]) for k, vals in EXAMPLE_REFSET_DICT.items()
)


class DirectMappingTest(unittest.TestCase):

    def test_example_gets_direct_mappings(self):
        res = preprocess_snomed.get_direct_refset_mapping(EXAMPLE_REFSET_DICT)
        self.assertEqual(res, EXPECTED_DIRECT_MAPPINGS)

    def test_example_w_extras_gets_direct_mappings(self):
        res = preprocess_snomed.get_direct_refset_mapping(EXAMPLE_REFSET_DICT_WITH_EXTRAS)
        self.assertEqual(res, EXPECTED_DIRECT_MAPPINGS)

    def test_example_no_priority_fails(self):
        with self.assertRaises(KeyError):
            preprocess_snomed.get_direct_refset_mapping(EXAMPLE_REFSET_DICT_NO_PRIORITY)

    def test_example_no_codfe_fails(self):
        with self.assertRaises(KeyError):
            preprocess_snomed.get_direct_refset_mapping(EXAMPLE_REFSET_DICT_NO_CODE)

EXAMPLE_SNOMED_PATH_OLD = "SnomedCT_InternationalRF2_PRODUCTION_20220831T120000Z"
EXAMPLE_SNOMED_PATH_NEW = "SnomedCT_UKClinicalRF2_PRODUCTION_20231122T000001Z"


class TestSnomedVersionsOPCS4(unittest.TestCase):

    def test_old_gets_old_OPCS4_mapping_nonuk_ext(self):
        snomed = preprocess_snomed.Snomed(EXAMPLE_SNOMED_PATH_OLD, uk_ext=False)
        self.assertEqual(snomed.opcs_refset_id, "1126441000000105")

    def test_old_gets_old_OPCS4_mapping_uk_ext(self):
        snomed = preprocess_snomed.Snomed(EXAMPLE_SNOMED_PATH_OLD, uk_ext=True)
        self.assertEqual(snomed.opcs_refset_id, "1126441000000105")

    def test_new_gets_new_OCPS4_mapping_uk_ext(self):
        snomed = preprocess_snomed.Snomed(EXAMPLE_SNOMED_PATH_NEW, uk_ext=True)
        self.assertEqual(snomed.opcs_refset_id, "1382401000000109")
