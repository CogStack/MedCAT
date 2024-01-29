import medcat.utils.spacy_compatibility as module_under_test
from medcat.utils.spacy_compatibility import _is_spacy_model_folder, _find_spacy_model_folder
from medcat.utils.spacy_compatibility import get_installed_spacy_version, get_installed_model_version
from medcat.utils.spacy_compatibility import _get_name_and_meta_of_spacy_model_in_medcat_modelpack
from medcat.utils.spacy_compatibility import get_name_and_version_of_spacy_model_in_medcat_modelpack
from medcat.utils.spacy_compatibility import _is_spacy_version_within_range
from medcat.utils.spacy_compatibility import medcat_model_pack_has_compatible_spacy_model
from medcat.utils.spacy_compatibility import is_older_spacy_version
from medcat.utils.spacy_compatibility import medcat_model_pack_has_semi_compatible_spacy_model

import unittest

from typing import Callable
import random
import string
import tempfile
import os
from contextlib import contextmanager


FAKE_SPACY_MODEL_NAME = "ff_core_fake_dr"
FAKE_SPACY_MODEL_DIR = os.path.join("tests", "resources", FAKE_SPACY_MODEL_NAME)
FAKE_MODELPACK_MODEL_DIR = os.path.join(FAKE_SPACY_MODEL_DIR, '..')


class SpacyModelFolderIdentifierTests(unittest.TestCase):
    expected_working_spacy_models = [
        "en_core_sci_sm",
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg",
        "en_core_web_trf",
        "nl_core_news_sm",
        "nl_core_news_md",
        "nl_core_news_lg",
    ]
    # the following were used in medcat models created prior
    # to v1.2.4
    expected_working_legacy_names = [
        "spacy_model"
    ]

    def test_works_expected_models(self):
        for model_name in self.expected_working_spacy_models:
            with self.subTest(model_name):
                self.assertTrue(_is_spacy_model_folder(model_name))

    def test_works_legacy_models(self):
        for model_name in self.expected_working_legacy_names:
            with self.subTest(model_name):
                self.assertTrue(_is_spacy_model_folder(model_name))

    def test_works_fill_path(self):
        for model_name in self.expected_working_legacy_names:
            full_folder_path = os.path.join("some", "folder", "structure", model_name)
            with self.subTest(full_folder_path):
                self.assertTrue(_is_spacy_model_folder(model_name))

    def get_all_garbage(self) -> list:
        """Generate garbage "spacy names".

        Returns:
            List[str]: Some random strings that shouldn't be spacy models.
        """
        my_examples = ["garbage_in_and_out", "meta_Presence", "something"]
        true_randoms_N10 = [''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) for _ in range(10)]
        true_randoms_N20 = [''.join(random.choices(string.ascii_uppercase + string.digits, k=20)) for _ in range(10)]
        return my_examples + true_randoms_N10 + true_randoms_N20

    def test_does_not_work_grabage(self):
        for garbage in self.get_all_garbage():
            with self.subTest(garbage):
                self.assertFalse(_is_spacy_model_folder(garbage))


class FindSpacyFolderJustOneFolderEmptyFilesTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls, spacy_folder_name='en_core_web_md') -> None:
        # setup temp folder
        cls.temp_folder = tempfile.TemporaryDirectory()
        cls.fake_modelpack_folder_name = cls.temp_folder.name
        # create spacy folder
        cls.spacy_folder = os.path.join(cls.fake_modelpack_folder_name, spacy_folder_name)
        os.makedirs(cls.spacy_folder)
        # create 2 empty files
        filenames = ["file1.dat", "file2.json"]
        filenames = [os.path.join(cls.fake_modelpack_folder_name, fn) for fn in filenames]
        for fn in filenames:
            with open(fn, 'w'):
                pass # open and write empty file
    
    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_folder.cleanup()

    def test_finds(self):
        found_folder_path = _find_spacy_model_folder(self.fake_modelpack_folder_name)
        self.assertEqual(found_folder_path, self.spacy_folder)


class FindSpacyFolderMoreFoldersEmptyFilesTests(FindSpacyFolderJustOneFolderEmptyFilesTests):

    @classmethod
    def setUpClass(cls, spacy_folder_name='en_core_web_md') -> None:
        super().setUpClass(spacy_folder_name)
        # add a few folders
        folder_names = ["meta_Presence", "garbage_in_garbage_out"]
        folder_names = [os.path.join(cls.fake_modelpack_folder_name, fn) for fn in folder_names]
        for folder in folder_names:
            os.makedirs(folder)


class SpacyVersionTests(unittest.TestCase):

    def test_version_received(self):
        installed = get_installed_spacy_version()
        import spacy
        expected = spacy.__version__
        self.assertEqual(installed, expected)


class InstalledVersionChecker(unittest.TestCase):

    def test_existing(self, model_name: str = 'en_core_web_md'):
        version = get_installed_model_version(model_name)
        self.assertIsInstance(version, str)
        self.assertNotEqual(version, "N/A")

    def test_non_existing(self, model_name: str = 'en_core_web_lg'):
        version = get_installed_model_version(model_name)
        self.assertIsInstance(version, str)
        self.assertEqual(version, "N/A")


class GetSpacyModelInfoTests(unittest.TestCase):
    expected_version = "3.1.0"

    @classmethod
    def setUpClass(cls) -> None:
        cls.name, cls.info = _get_name_and_meta_of_spacy_model_in_medcat_modelpack(FAKE_MODELPACK_MODEL_DIR)

    def test_reads_name(self):
        self.assertEqual(self.name, FAKE_SPACY_MODEL_NAME)

    def test_reads_info(self):
        self.assertIsInstance(self.info, dict)
        self.assertTrue(self.info)  # not empty


class GetSpacyModelVersionTests(GetSpacyModelInfoTests):
    expected_spacy_version = ">=3.1.0,<4.0.0"

    @classmethod
    def setUpClass(cls) -> None:
        (cls.name,
         cls.version,
         cls.spacy_version) = get_name_and_version_of_spacy_model_in_medcat_modelpack(FAKE_MODELPACK_MODEL_DIR)

    def test_name_correct(self):
        self.assertEqual(self.name, FAKE_SPACY_MODEL_NAME)

    def test_version_correct(self):
        self.assertEqual(self.version, self.expected_version)

    def test_spacy_version_correct(self):
        self.assertEqual(self.spacy_version, self.expected_spacy_version)


@contextmanager
def custom_spacy_version(mock_version: str):
    """Changes the apparently installed spacy version.
    """
    print(f"Mocking spacy version to: {mock_version}")
    _old_method = module_under_test.get_installed_spacy_version
    module_under_test.get_installed_spacy_version = lambda: mock_version
    yield mock_version
    print("Returning regular spacy version getter")
    module_under_test.get_installed_spacy_version = _old_method


class VersionMockBaseTests(unittest.TestCase):

    def base_subtest_for(self, target_fun: Callable[[str], bool],
                     spacy_model_range: str, spacy_version: str, should_work: bool) -> None:
        with self.subTest(spacy_version):
            if should_work:
                self.assertTrue(target_fun(spacy_model_range))
            else:
                self.assertFalse(target_fun(spacy_model_range))

    def base_check_version(self, target_fun: Callable[[str], bool],
                       spacy_model_range: str, spacy_version: str, should_work: bool = True) -> None:
        with custom_spacy_version(spacy_version):
            self.base_subtest_for(target_fun, spacy_model_range, spacy_version, should_work)

class SpacyVersionMockBaseTests(VersionMockBaseTests):

    def _subtest_for(self, spacy_model_range: str, spacy_version: str, should_work: bool) -> None:
        return self.base_subtest_for(_is_spacy_version_within_range,
                                    spacy_model_range, spacy_version, should_work)

    def _check_version(self, spacy_model_range: str, spacy_version: str, should_work: bool = True) -> None:
        return self.base_check_version(_is_spacy_version_within_range,
                                      spacy_model_range, spacy_version, should_work)


class SpacyVersionInRangeOldRangeTests(SpacyVersionMockBaseTests):
    """This is for versions before 1.7.0.
    Those versions used to have spacy constraints of 'spacy<3.1.4,>=3.1.0'
    and as such, they used v3.1.0 of en_core_web_md.
    """
    spacy_model_range = ">=3.1.0,<3.2.0"  # model range for en_core_web_md-3.1.0
    useful_spacy_versions = ["3.1.0", "3.1.2", "3.1.3"]
    unsupported_spacy_versions = ["3.2.0", "3.5.3", "3.6.0"]

    def test_works_in_range(self):
        for spacy_version in self.useful_spacy_versions:
            self._check_version(self.spacy_model_range, spacy_version, should_work=True)

    def test_not_suitable_outside_range(self):
        for spacy_version in self.unsupported_spacy_versions:
            self._check_version(self.spacy_model_range, spacy_version, should_work=False)


class SpacyVersionInRangeNewRangeTests(SpacyVersionInRangeOldRangeTests):
    """This is for versions AFTER (and includring) 1.7.0.
    Those versions used to have spacy constraints of 'spacy>=3.1.0'
    and as such, we use v3.4.0 of en_core_web_md.

    In this setup, generally (in GHA at 14.12.2023)
    the spacy version for python version:
        3.8  -> spacy-3.7.2
        3.9  -> spacy-3.7.2
        3.10 -> spacy-3.7.2
        3.11 -> spacy-3.7.2
    Alongside the `en_core_web_md-3.4.0` is installed.
    It technically has the compatibility of >=3.4.0,<3.5.0.
    But practically, I've seen no issues with spacy==3.7.2.
    """
    spacy_model_range = ">=3.1.0"  # model range for medcat>=1.7.0
    useful_spacy_versions = ["3.1.0", "3.1.2", "3.1.3",
                             "3.7.2", "3.6.3"]
    unsupported_spacy_versions = ["3.0.0"]


class ModelPackHasCompatibleSpacyRangeTests(unittest.TestCase):
    test_spacy_version = "3.1.0"

    def test_is_in_range(self):
        with custom_spacy_version(self.test_spacy_version):
            b = medcat_model_pack_has_compatible_spacy_model(FAKE_MODELPACK_MODEL_DIR)
            self.assertTrue(b)

class ModelPackHasInCompatibleSpacyRangeTests(unittest.TestCase):
    test_spacy_version = "3.0.0"

    def test_is_not_in_range(self):
        with custom_spacy_version(self.test_spacy_version):
            b = medcat_model_pack_has_compatible_spacy_model(FAKE_MODELPACK_MODEL_DIR)
            self.assertFalse(b)


class IsOlderSpacyVersionTests(VersionMockBaseTests):
    test_spacy_version = "3.4.4"
    expected_older = ["3.1.0", "3.2.0", "3.3.0", "3.4.0"]
    expected_newer = ["3.5.0", "3.6.0", "3.7.1"]

    def _check_version(self, model_version: str, should_work: bool = True) -> None:
        self.base_check_version(is_older_spacy_version, model_version, self.test_spacy_version, should_work)

    def test_older_works(self):
        for model_version in self.expected_older:
            self._check_version(model_version, should_work=True)

    def test_newer_fails(self):
        for model_version in self.expected_newer:
            self._check_version(model_version, should_work=False)


class HasSemiCompatibleSpacyModelTests(unittest.TestCase):
    # model version on file is 3.1.0,
    # and spacy_version range >=3.1.0,<3.2.0"
    good_spacy_version = "3.1.3"
    semi_good_spacy_version = "3.4.4"  # newer than the model
    bad_spacy_version = "3.0.0"  # older than the model

    def run_subtest(self, spacy_version: str, should_work: bool) -> None:
        with custom_spacy_version(spacy_version):
            if should_work:
                self.assertTrue(medcat_model_pack_has_semi_compatible_spacy_model(FAKE_MODELPACK_MODEL_DIR))
            else:
                self.assertFalse(medcat_model_pack_has_semi_compatible_spacy_model(FAKE_MODELPACK_MODEL_DIR))

    def test_works_compatible_spacy_version(self):
        self.run_subtest(self.good_spacy_version, should_work=True)

    def test_works_semi_compatible_spacy_version(self):
        self.run_subtest(self.semi_good_spacy_version, should_work=True)

    def test_fails_incompatible_spacy_version(self):
        self.run_subtest(self.bad_spacy_version, should_work=False)
