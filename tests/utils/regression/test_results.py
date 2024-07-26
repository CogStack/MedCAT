
from typing import Optional
import unittest

from medcat.utils.regression.targeting import TranslationLayer
from medcat.utils.regression.results import Finding, MalformedFinding
from medcat.utils.regression.results import FindingDeterminer

from .test_checking import FakeCDB


def _determine_raw_helper(exp_start: int, exp_end: int,
                            start: int, end: int,
                            strict_only: bool = False) -> Optional[Finding]:
    return FindingDeterminer("NO_MATTER", exp_start, exp_end, None,
                                None, strict_only=strict_only)._determine_raw(start, end)


class FindingRawTests(unittest.TestCase):
    EXAMPLES = [
        # (exp start, exp end, start, end), expected finding
        # start < exp_start
        ((10, 15, 0, 1), None),
        ((10, 15, 0, 11), Finding.PARTIAL_OVERLAP),
        ((10, 15, 0, 15), Finding.BIGGER_SPAN_LEFT),
        ((10, 15, 0, 25), Finding.BIGGER_SPAN_BOTH),
        # start == exp_start
        ((10, 15, 10, 12), Finding.SMALLER_SPAN),
        ((10, 15, 10, 15), Finding.IDENTICAL),
        ((10, 15, 10, 25), Finding.BIGGER_SPAN_RIGHT),
        # exp_start < start < exp_end
        ((10, 15, 12, 13), Finding.SMALLER_SPAN),
        ((10, 15, 12, 15), Finding.SMALLER_SPAN),
        ((10, 15, 12, 25), Finding.PARTIAL_OVERLAP),
        # exp_start >= end_end
        ((10, 15, 20, 25), None),
    ]

    def test_finds_correctly(self):
        for args, expected in self.EXAMPLES:
            with self.subTest(f"With args {args}"):
                found = _determine_raw_helper(*args)
                self.assertEqual(found, expected)

    def test_exception_when_improper_start_end(self):
        with self.assertRaises(MalformedFinding):
            _determine_raw_helper(0, 1, 10, 0)

    def test_exception_when_improper_expected_start_end(self):
        with self.assertRaises(MalformedFinding):
            _determine_raw_helper(10, 1, 0, 1)


def _get_example_ent(cui: str = "CUI1", start: int = 10, end: int = 15):
    return {"cui": cui,
            "start": start,
            "end": end}


def _get_example_kwargs(cui: str = "CUI1",
                        exp_start: int = 10, exp_end: int = 15):
    return {
        "exp_cui": cui,
        "exp_start": exp_start,
        "exp_end": exp_end,
        "check_children": True,
        "check_parent": True,
        "check_grandparent": True
        }


class FindingFromEntsTests(unittest.TestCase):
    EXAMPLES = [
        # identical
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent()}}, Finding.IDENTICAL),
        # bigger span
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=5)},
         }, Finding.BIGGER_SPAN_LEFT),
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(end=25)},
         }, Finding.BIGGER_SPAN_RIGHT),
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=5, end=25)},
         }, Finding.BIGGER_SPAN_BOTH),
        # smaller span
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(end=13)},
         }, Finding.SMALLER_SPAN),
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(end=13)},
         }, Finding.SMALLER_SPAN),
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=11, end=13)},
         }, Finding.SMALLER_SPAN),
        # overlapping span
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=5, end=12)},
         }, Finding.PARTIAL_OVERLAP),
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=12, end=122)},
         }, Finding.PARTIAL_OVERLAP),
        # identical with some noise start
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=0, end=5),
                             1: _get_example_ent()},
         }, Finding.IDENTICAL),
        # identical with some noise end
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(),
                             1: _get_example_ent(start=20, end=25)},
         }, Finding.IDENTICAL),
        # identical with some noise both sides
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=0, end=5),
                             1: _get_example_ent(),
                             2: _get_example_ent(start=20, end=25)},
         }, Finding.IDENTICAL),
        # FAILURES
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(cui="CUI2")},
         }, Finding.FAIL),
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=0, end=5)},
         }, Finding.FAIL),
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=20, end=25)},
         }, Finding.FAIL),
    ]
    NR_OF_EXAMPLES = len(EXAMPLES)
    TL = TranslationLayer.from_CDB(FakeCDB())

    def test_finds_examples(self):
        self.assertEqual(len(self.EXAMPLES), self.NR_OF_EXAMPLES)
        for nr, (ekwargs, expected) in enumerate(self.EXAMPLES):
            with self.subTest(f"With [{nr}] kwargs {ekwargs}"):
                found = Finding.determine(tl=self.TL, **ekwargs)
                self.assertEqual(found, expected)


class FindingFromEntsStrictTests(FindingFromEntsTests):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.EXAMPLES = [
            ({**e_kwargs, 'strict_only': True}, e_exp) for e_kwargs, e_exp in cls.EXAMPLES.copy()
            if e_exp in (Finding.IDENTICAL, Finding.FAIL)
        ]
        cls.NR_OF_EXAMPLES = len(cls.EXAMPLES)
        cls.FAIL_EXAMPLES = [
            ({**e_kwargs, 'strict_only': True}, e_exp) for e_kwargs, e_exp in cls.EXAMPLES.copy()
            if e_exp not in (Finding.IDENTICAL, Finding.FAIL)
        ]

    def test_fails_on_non_identical_or_fail_in_strict_mode(self):
        for nr, (ekwargs, _) in enumerate(self.FAIL_EXAMPLES):
            with self.subTest(f"With [{nr}] kwargs {ekwargs}"):
                found = Finding.determine(tl=self.TL, **ekwargs)
                self.assertEqual(found, Finding.FAIL)
