import unittest
import yaml
from medcat.utils.regression.checking import RegressionChecker

from medcat.utils.regression.converting import medcat_export_json_to_regression_yml


class TestConversion(unittest.TestCase):
    def_file_name = 'tests/resources/medcat_trainer_export.json'

    def test_conversion_default_gets_str(self):
        converted_yaml = medcat_export_json_to_regression_yml(
            self.def_file_name)
        self.assertIsInstance(converted_yaml, str)
        self.assertGreater(len(converted_yaml), 0)

    def test_conversion_default_gets_yml(self):
        converted_yaml = medcat_export_json_to_regression_yml(
            self.def_file_name)
        d = yaml.safe_load(converted_yaml)
        self.assertIsInstance(d, dict)
        self.assertGreater(len(d), 0)

    def test_conversion_valid_regression_case(self):
        converted_yaml = medcat_export_json_to_regression_yml(
            self.def_file_name)
        d = yaml.safe_load(converted_yaml)
        checker = RegressionChecker.from_dict(d)
        self.assertIsInstance(checker, RegressionChecker)
