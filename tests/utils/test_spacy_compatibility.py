from medcat.utils.spacy_compatibility import is_spacy_model_folder

import unittest

import random
import string


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
