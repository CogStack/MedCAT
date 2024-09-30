from functools import partial
import os
from json import load as load_json
from enum import Enum, auto

from unittest import TestCase

from medcat.utils.regression import utils
from medcat.utils.regression.checking import RegressionSuite

from medcat.utils.normalizers import get_all_edits_n


class PartialSubstituationTests(TestCase):
    TEXT1 = "This [PH1] has one placeholder"
    PH1 = "PH1"
    REPLACEMENT1 = "<REPLACE1>"

    def test_fails_with_1_ph(self):
        with self.assertRaises(utils.IncompatiblePhraseException):
            utils.partial_substitute(self.TEXT1, self.PH1, self.REPLACEMENT1, 0)

    TEXT2 = "This [PH1] has [PH1] multiple (2) placeholders"

    def assert_is_correct_for_regr(self, text: str, placeholder: str):
        # should leave a placeholder in
        self.assertIn(placeholder, text)
        # and only 1
        self.assertEqual(text.count(placeholder), 1)

    def assert_has_replaced_and_is_suitable(self, text: str, placeholder: str, replacement: str,
                                            repl_count: int):
        self.assert_is_correct_for_regr(text, placeholder)
        self.assertIn(replacement, text)
        self.assertEqual(text.count(replacement), repl_count)

    def test_works_with_2_ph_0th(self):
        text = utils.partial_substitute(self.TEXT2, self.PH1, self.REPLACEMENT1, 0)
        self.assert_has_replaced_and_is_suitable(text, self.PH1, self.REPLACEMENT1, 1)

    def test_works_with_2_ph_1st(self):
        text = utils.partial_substitute(self.TEXT2, self.PH1, self.REPLACEMENT1, 1)
        self.assert_has_replaced_and_is_suitable(text, self.PH1, self.REPLACEMENT1, 1)

    def test_fails_if_too_high_a_change_nr(self):
        with self.assertRaises(utils.IncompatiblePhraseException):
            utils.partial_substitute(self.TEXT1, self.PH1, self.REPLACEMENT1, 2)

    TEXT3 = "No [PH1] is [PH1] safe [PH1] eh"

    def test_work_with_3_ph(self):
        for nr in range(self.TEXT3.count(self.PH1)):
            with self.subTest(f"Placeholder #{nr}"):
                text = utils.partial_substitute(self.TEXT3, self.PH1, self.REPLACEMENT1, nr)
                self.assert_has_replaced_and_is_suitable(text, self.PH1, self.REPLACEMENT1, 2)

    def test_all_possibilities_are_similar(self):
        texts = [utils.partial_substitute(self.TEXT3, self.PH1, self.REPLACEMENT1, nr)
                 for nr in range(self.TEXT3.count(self.PH1))]
        # they should all have the same length
        lengths = [len(t) for t in texts]
        self.assertTrue(all(cl == lengths[0] for cl in lengths))
        # they should all have the same character composition
        # i.e they should compose of the same exact characters
        char_compos = [set(t) for t in texts]
        self.assertTrue(all(cchars == char_compos[0] for cchars in char_compos))
        # and there should be the same amount for each as well
        char_counts = [{c: t.count(c) for c in char_compos[0]} for t in texts]
        self.assertTrue(all(cchars == char_counts[0] for cchars in char_counts))    


class StringLengthLimiterTests(TestCase):
    short_str = "short str"
    max_len = 25
    keep_front = max_len // 2 - 3
    keep_rear = max_len // 2 - 3
    long_str = " ".join([short_str] * 10)
    limiter = partial(utils.limit_str_len, max_length=max_len,
                      keep_front=keep_front, keep_rear=keep_rear)

    @classmethod
    def setUpClass(cls) -> None:
        cls.got_short = cls.limiter(cls.short_str)
        cls.got_long = cls.limiter(cls.long_str)

    def test_leaves_short(self):
        self.assertEqual(self.short_str, self.got_short)

    def test_changes_long(self):
        self.assertNotEqual(self.long_str, self.got_long)

    def test_long_gets_shorter(self):
        self.assertGreater(self.long_str, self.got_long)

    def test_long_includes_chars(self, chars: str = 'chars'):
        self.assertNotIn(chars, self.long_str)
        self.assertIn(chars, self.got_long)

    def test_keeps_max_length(self):
        s = self.got_long[:self.max_len]
        self.assertEqual(s, self.limiter(s))

    def test_does_not_keep_1_longer_than_max_lenght(self):
        s = self.got_long[:self.max_len + 1]
        self.assertNotEqual(s, self.limiter(s))


class MCTExportConverterTests(TestCase):
    MCT_EXPORT_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                                   'resources', 'medcat_trainer_export.json')

    @classmethod
    def setUpClass(cls) -> None:
        with open(cls.MCT_EXPORT_PATH) as f:
            cls.mct_export = load_json(f)
        cls.converter = utils.MedCATTrainerExportConverter(cls.mct_export)
        cls.converted = cls.converter.convert()
        cls.rc = RegressionSuite.from_dict(cls.converted, name="TEST SUITE 3")

    def test_converted_is_dict(self):
        self.assertIsInstance(self.converted, dict)

    def test_converted_can_build(self):
        self.assertIsInstance(self.rc, RegressionSuite)

    def test_converted_is_nonempty(self):
        self.assertGreater(len(self.rc.cases), 0)
        self.assertGreater(self.rc.estimate_total_distinct_cases(), 0)


class MyE1(Enum):
    """Has class doc-string"""
    A1 = auto()
    """A1 doc string"""
    A2 = auto()
    """A2 doc string"""


class MyE2(Enum): # no class-level doc string
    A1 = auto()
    """A1 doc string"""
    A2 = auto()
    """A2 doc string"""


class MyE3(Enum):  # this will not be changed
    """The CLASS-specific doc string"""
    A1 = auto()
    """A1 doc string"""
    A2 = auto()
    """A2 doc string"""


class EnumDocStringCapturingClass(TestCase):

    @classmethod
    def get_doc_string(cls, cnst: Enum) -> str:
        # NOTE: this assumes the doc strings are built in this format
        return cnst.name + " doc string"

    @classmethod
    def setUpClass(cls) -> None:
        utils.add_doc_strings_to_enum(MyE1)
        utils.add_doc_strings_to_enum(MyE2)

    def assert_has_doc_strings(self, cls):
        for ec in cls:
            with self.subTest(str(ec)):
                self.assertEqual(ec.__doc__, self.get_doc_string(ec))

    def test_class_w_class_docstring_gets_doc_strings(self):
        self.assert_has_doc_strings(MyE1)

    def test_class_wo_class_docstring_gets_doc_strings(self):
        self.assert_has_doc_strings(MyE2)

    def test_unchanged_does_not_have_correct_doc_strings(self):
        for ec in MyE3:
            with self.subTest(str(ec)):
                self.assertNotEqual(ec.__doc__, self.get_doc_string(ec))

    def test_unchanged_has_class_doc_Strings(self):
        for ec in MyE3:
            with self.subTest(str(ec)):
                self.assertEqual(ec.__doc__, MyE3.__doc__)


class EditBaseTests(TestCase):
    WORDS = ['WORDs', 'multi word', 'long-ass-word',
             'complexsuperlongwordthatexists']


class EditEstimationTests(EditBaseTests):

    def assert_can_estimate_dist(self, edit_distance: int, tol_perc: float):
        for word in self.WORDS:
            with self.subTest(word):
                got = len(list(get_all_edits_n(word, False, edit_distance)))
                expected = utils.estimate_num_variants(len(word), edit_distance)
                ratio = got / expected
                self.assertTrue(1 - tol_perc < ratio < 1 + tol_perc,
                                f"Ratio {ratio} vs TOL {tol_perc}")

    def test_can_estimate_dist1(self):
        self.assert_can_estimate_dist(1, 0.04)

    def test_can_estimate_dist2(self):
        self.assert_can_estimate_dist(2, 0.06)


class EditTests(EditBaseTests):
    ORIG_WORD = "WORD"
    EDIT_DIST = 1
    ALL_EDITS = list(get_all_edits_n(ORIG_WORD, False, EDIT_DIST))
    LEN = len(ALL_EDITS)
    # NOTE: can't use the full length since the estimation is lower
    PICKS = [1, 5, 10, LEN - 20]
    RNG_SEED = 42

    def test_can_pick_correct_number(self):
        for pick in self.PICKS:
            with self.subTest(f"Pick {pick}"):
                picked = list(utils.pick_random_edits(
                    self.ALL_EDITS, edit_distance=self.EDIT_DIST,
                    num_to_pick=pick, orig_len=len(self.ORIG_WORD),
                    rng_seed=self.RNG_SEED))
                self.assertEqual(len(picked), pick)
                # make sure the names are unique
                self.assertEqual(len(picked), len(set(picked)))
