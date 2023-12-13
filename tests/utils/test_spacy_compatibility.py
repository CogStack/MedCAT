from medcat.utils.spacy_compatibility import is_spacy_model_folder, find_spacy_model_folder
from medcat.utils.spacy_compatibility import get_installed_spacy_version

import unittest

import random
import string
import tempfile
import os


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
                self.assertTrue(is_spacy_model_folder(model_name))

    def test_works_legacy_models(self):
        for model_name in self.expected_working_legacy_names:
            with self.subTest(model_name):
                self.assertTrue(is_spacy_model_folder(model_name))

    def test_works_fill_path(self):
        for model_name in self.expected_working_legacy_names:
            full_folder_path = os.path.join("some", "folder", "structure", model_name)
            with self.subTest(full_folder_path):
                self.assertTrue(is_spacy_model_folder(model_name))

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
                self.assertFalse(is_spacy_model_folder(garbage))


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
        found_folder_path = find_spacy_model_folder(self.fake_modelpack_folder_name)
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
