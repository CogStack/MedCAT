import json
import unittest
import yaml

from medcat.utils.regression.checking import RegressionChecker

from medcat.utils.regression.converting import medcat_export_json_to_regression_yml
from medcat.utils.regression.targeting import TargetInfo


class FakeTranslationLayer:

    def __init__(self, mct_export: dict) -> None:
        self.mct_export = json.loads(mct_export)

    def all_targets(self):  # -> Iterator[TargetInfo]:
        for project in self.mct_export['projects']:
            for doc in project['documents']:
                for ann in doc['annotations']:
                    yield TargetInfo(ann['cui'], ann['value'])


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

    def test_correct_number_of_cases(self):
        converted_yaml = medcat_export_json_to_regression_yml(
            self.def_file_name)
        checker = RegressionChecker.from_dict(yaml.safe_load(converted_yaml))
        with open(self.def_file_name, 'r') as f:
            mct_export = f.read()
            expected = mct_export.count('"cui":')
        nr_of_cases = len(list(checker.get_all_subcases(
            FakeTranslationLayer(mct_export))))
        self.assertEqual(nr_of_cases, expected)
