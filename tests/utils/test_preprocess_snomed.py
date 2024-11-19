import os
from typing import Dict
import contextlib

from medcat.utils import preprocess_snomed

import unittest
from unittest.mock import patch


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
EXAMPLE_SNOMED_PATH_OLD_UK = "SnomedCT_UKClinicalRF2_PRODUCTION_20220831T120000Z"
EXAMPLE_SNOMED_PATH_NEW = "SnomedCT_UKClinicalRF2_PRODUCTION_20231122T000001Z"


@contextlib.contextmanager
def patch_fake_files(path: str, subfiles: list = [],
                     subdirs: list = ["Snapshot"]):
    def cur_listdir(file_path: str, *args, **kwargs) -> list:
        if file_path == path:
            return subfiles + subdirs
        for sd in subdirs:
            subdir = os.path.join(path, sd)
            if subdir == path:
                return []
        raise FileNotFoundError(path)

    def cur_isfile(file_path: str, *args, **kwargs) -> bool:
        print("CUR isfile", file_path)
        return file_path == path or file_path in [os.path.join(path, subfiles)]

    def cur_isdir(file_path: str, *args, **kwrags) -> bool:
        print("CUR isdir", file_path)
        return file_path == path or file_path in [os.path.join(path, subdirs)]

    with patch("os.listdir", new=cur_listdir):
        with patch("os.path.isfile", new=cur_isfile):
            with patch("os.path.isdir", new=cur_isdir):
                yield


class TestSnomedVersionsOPCS4(unittest.TestCase):

    def test_old_gets_old_OPCS4_mapping(self):
        with patch_fake_files(EXAMPLE_SNOMED_PATH_OLD):
            snomed = preprocess_snomed.Snomed(EXAMPLE_SNOMED_PATH_OLD)
        snomed._set_extension(snomed._determine_release(EXAMPLE_SNOMED_PATH_OLD),
                              snomed._determine_extension(EXAMPLE_SNOMED_PATH_OLD))
        self.assertEqual(snomed.opcs_refset_id, "1382401000000109")  # defaults to this now

    def test_old_gets_old_OPCS4_mapping_UK(self):
        with patch_fake_files(EXAMPLE_SNOMED_PATH_OLD_UK):
            snomed = preprocess_snomed.Snomed(EXAMPLE_SNOMED_PATH_OLD_UK)
        snomed._set_extension(snomed._determine_release(EXAMPLE_SNOMED_PATH_OLD_UK),
                              snomed._determine_extension(EXAMPLE_SNOMED_PATH_OLD_UK))
        self.assertEqual(snomed.opcs_refset_id, "1126441000000105")

    def test_new_gets_new_OCPS4_mapping(self):
        with patch_fake_files(EXAMPLE_SNOMED_PATH_NEW):
            snomed = preprocess_snomed.Snomed(EXAMPLE_SNOMED_PATH_NEW)
        snomed._set_extension(snomed._determine_release(EXAMPLE_SNOMED_PATH_NEW),
                              snomed._determine_extension(EXAMPLE_SNOMED_PATH_NEW))
        self.assertEqual(snomed.opcs_refset_id, "1382401000000109")


class TestSnomedModelGetter(unittest.TestCase):
    WORKING_BASE_NAMES = [
        "SnomedCT_InternationalRF2_PRODUCTION_20240201T120000Z",
        "SnomedCT_InternationalRF2_PRODUCTION_20240601T120000Z",
        "SnomedCT_UKClinicalRF2_PRODUCTION_20240410T000001Z",
        "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION_20240410T000001Z",
        "SnomedCT_UKDrugRF2_PRODUCTION_20240508T000001Z",
        "SnomedCT_UKEditionRF2_PRODUCTION_20240410T000001Z",
        "SnomedCT_UKEditionRF2_PRODUCTION_20240508T000001Z",
        "SnomedCT_Release_AU1000036_20240630T120000Z",
    ]
    FAILING_BASE_NAMES = [
        "uk_sct2cl_38.2.0_20240605000001Z",
        "uk_sct2cl_32.6.0_20211027000001Z",
    ]
    PATH = os.path.join("path", "to", "release")

    def _pathify(self, in_list: list) -> list:
        return [os.path.join(self.PATH, folder) for folder in in_list]

    def assert_got_version(self, snomed: preprocess_snomed.Snomed, raw_name: str):
        rel_list = snomed.snomed_releases
        self.assertIsInstance(rel_list, list)
        self.assertEqual(len(rel_list), 1)
        rel = rel_list[0]
        self.assertIsInstance(rel, str)
        self.assertIn(rel, raw_name)
        self.assertEqual(rel, raw_name[-16:-8])

    def assert_all_work(self, all_paths: list):
        for path in all_paths:
            with self.subTest(f"Rrelease name: {path}"):
                with patch_fake_files(path):
                    snomed = preprocess_snomed.Snomed(path)
                self.assert_got_version(snomed, path)

    def test_gets_model_form_basename(self):
        self.assert_all_work(self.WORKING_BASE_NAMES)

    def test_gets_model_from_path(self):
        full_paths = self._pathify(self.WORKING_BASE_NAMES)
        self.assert_all_work(full_paths)

    def assert_raises(self, folder_path: str):
        with self.assertRaises(preprocess_snomed.UnkownSnomedReleaseException):
            preprocess_snomed.Snomed._determine_release(folder_path, strict=True)

    def assert_all_raise(self, folder_paths: list):
        for folder_path in folder_paths:
            with self.subTest(f"Folder: {folder_path}"):
                self.assert_raises(folder_path)

    def test_fails_on_incorrect_names_strict(self):
        self.assert_all_raise(self.FAILING_BASE_NAMES)

    def test_fails_on_incorrect_paths_strict(self):
        full_paths = self._pathify(self.FAILING_BASE_NAMES)
        self.assert_all_raise(full_paths)

    def assert_all_get_no_version(self, folder_paths: list):
        for folder_path in folder_paths:
            with self.subTest(f"Folder: {folder_path}"):
                det_rel = preprocess_snomed.Snomed._determine_release(folder_path, strict=False)
                self.assertEqual(det_rel, preprocess_snomed.Snomed.NO_VERSION_DETECTED)

    def test_gets_no_version_incorrect_names_nonstrict(self):
        self.assert_all_get_no_version(self.FAILING_BASE_NAMES)

    def test_gets_no_version_incorrect_paths_nonstrict(self):
        full_paths = self._pathify(self.FAILING_BASE_NAMES)
        self.assert_all_get_no_version(full_paths)
