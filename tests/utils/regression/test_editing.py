
import unittest
import yaml

from medcat.utils.regression.checking import RegressionChecker

from medcat.utils.regression.editing import combine_contents


class TestCombining(unittest.TestCase):
    tests1 = """
test-case-1: 
  targeting:
    filters:
      NAME: tcn1
  phrases:
    - Some %s phrase
    """.strip()
    tests1_cp = """
test-case-1: 
  targeting:
    filters:
      NAME: tcn1-cp
  phrases:
    - Some %s phrase
    """.strip()
    tests2 = """
test-case-2: 
  targeting:
    filters:
      NAME: tcn2
  phrases:
    - Some %s phrase
    """.strip()
    tests2_cp = """
test-case-2: 
  targeting:
    filters:
      NAME: tcn2-cp
  phrases:
    - Some %s phrase
    """.strip()

    def assert_simple_combination(self, one: str, two: str, combined: str = None,
                                  expect_addition: bool = True,
                                  check_str_len: bool = True,
                                  ignore_identicals: bool = True) -> str:
        if not combined:
            combined = combine_contents(
                one, two, ignore_identicals=ignore_identicals)
        self.assertIsInstance(combined, str)
        c1 = RegressionChecker.from_dict(yaml.safe_load(one))
        c2 = RegressionChecker.from_dict(yaml.safe_load(two))
        cc = RegressionChecker.from_dict(yaml.safe_load(combined))
        nc1, nc2, ncc = len(c1.cases), len(c2.cases), len(cc.cases)
        if expect_addition:
            self.assertEqual(ncc, nc1 + nc2)
        else:
            # total must be greater or equal than the max
            self.assertGreaterEqual(ncc, max(nc1, nc2))
        if check_str_len:
            self.assertGreater(len(combined), len(one))
            self.assertGreater(len(combined), len(two))
            # print(f'From\n{one}\nand\n{two}\nto\n{combined}')
            # account for a newline in the middle
            if expect_addition:
                self.assertGreaterEqual(len(combined), len(one) + len(two))
        return combined

    def test_combining_makes_longer_yaml(self):
        self.assert_simple_combination(self.tests1, self.tests2)

    def test_combinig_renames_similar_case(self):
        self.assert_simple_combination(self.tests1, self.tests1_cp)

    def test_combining_combines(self):
        # print('\n\nin adding new case\n\n')
        combined = self.assert_simple_combination(
            self.tests1, self.tests1, expect_addition=False, ignore_identicals=False)
        cc = RegressionChecker.from_dict(yaml.safe_load(combined))
        # print('\n\nEND test_combining_combines')
        self.assertEqual(len(cc.cases), 1)
        self.assertEqual(len(cc.cases[0].phrases), 2)

    def test_combining_no_combine_when_ignoring_identicals(self):
        # print('\n\nin adding new case\n\n')
        combined = self.assert_simple_combination(
            self.tests1, self.tests1, expect_addition=False, ignore_identicals=True)
        cc = RegressionChecker.from_dict(yaml.safe_load(combined))
        # print('\n\nEND test_combining_combines')
        self.assertEqual(len(cc.cases), 1)
        self.assertEqual(len(cc.cases[0].phrases), 1)
