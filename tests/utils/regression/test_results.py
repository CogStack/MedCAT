
from typing import Optional
import unittest

from medcat.utils.regression.targeting import TranslationLayer
from medcat.utils.regression.results import FailDescriptor, FailReason
from medcat.utils.regression.results import Finding, MalformedFinding
from medcat.utils.regression.results import FindingDeterminer

from .test_checking import FakeCDB


class TestFailReason(unittest.TestCase):
    cui2names = {
        'cui1': set(['name-cui1-1', 'name-cui1-2']),
        'cui2': set(['name-cui2-1', 'name-cui2-2']),
        'cui3': set(['name-cui3-1', 'name-cui3-2', 'name-cui3-3']),
        'cui4': set(['name-cui4-1', ]),
    }
    # only works if one name corresponds to one CUI
    name2cuis = dict([(name, set([cui]))
                     for cui, names in cui2names.items() for name in names])
    cui2type_ids = {
        'cui1': set(['T1', ]),
        'cui2': set(['T1', ]),
        'cui3': set(['T2', ]),
        'cui4': set(['T4', ])
    }
    cui2children = {}  # none for now
    tl = TranslationLayer(cui2names, name2cuis, cui2type_ids, cui2children)

    def test_cui_not_found(self, cui='cui-100', name='random n4m3'):
        fr = FailDescriptor.get_reason_for(cui, name, {}, self.tl)
        self.assertIs(fr.reason, FailReason.CUI_NOT_FOUND)

    def test_cui_name_found(self, cui='cui1', name='random n4m3-not-there'):
        fr = FailDescriptor.get_reason_for(cui, name, {}, self.tl)
        self.assertIs(fr.reason, FailReason.NAME_NOT_FOUND)


class TestFailReasonWithResultAndChildren(TestFailReason):
    res_w_cui1 = {'entities': {
        # cui1
        1: {'source_value': list(TestFailReason.cui2names['cui1'])[0], 'cui': 'cui1'},
    }}
    res_w_cui2 = {'entities': {
        # cui2
        1: {'source_value': list(TestFailReason.cui2names['cui2'])[0], 'cui': 'cui2'},
    }}
    res_w_both = {'entities': {
        # cui1
        1: {'source_value': list(TestFailReason.cui2names['cui1'])[0], 'cui': 'cui1'},
        # cui2
        2: {'source_value': list(TestFailReason.cui2names['cui2'])[0], 'cui': 'cui2'},
    }}
    cui2children = {'cui1': set(['cui2'])}
    tl = TranslationLayer(TestFailReason.cui2names, TestFailReason.name2cuis,
                          TestFailReason.cui2type_ids, cui2children)

    def test_found_child(self, cui='cui1', name='name-cui1-2'):
        fr = FailDescriptor.get_reason_for(cui, name, self.res_w_cui2, self.tl)
        self.assertIs(fr.reason, FailReason.CUI_CHILD_FOUND)

    def test_found_parent(self, cui='cui2', name='name-cui2-1'):
        fr = FailDescriptor.get_reason_for(cui, name, self.res_w_cui1, self.tl)
        self.assertIs(fr.reason, FailReason.CUI_PARENT_FOUND)


class TestFailReasonWithSpanningConcepts(unittest.TestCase):
    cui2names = {
        'cui1': ('shallow', 'shallow2'),
        'cui1.1': ('broader shallow', 'broader shallow2'),
        'cui1.1.1': ('even broader shallow', 'even broader shallow2'),
        'cui2': ('name-2', ),
    }
    # only works if one name corresponds to one CUI
    name2cuis = dict([(name, set([cui]))
                     for cui, names in cui2names.items() for name in names])
    cui2type_ids = {
        'cui1': set(['T1', ]),
        'cui1.1': set(['T1', ]),
        'cui1.1.1': set(['T1', ])
    }
    cui2children = {}  # none for now
    tl = TranslationLayer(cui2names, name2cuis, cui2type_ids, cui2children)

    res_w_cui1 = {'entities': {
        # cui1
        1: {'source_value': list(cui2names['cui1'])[0], 'cui': 'cui1'},
    }}

    res_w_cui11 = {'entities': {
        # cui1.1
        1: {'source_value': list(cui2names['cui1.1'])[0], 'cui': 'cui1.1'},
    }}

    res_w_cui111 = {'entities': {
        # cui1.1.1
        1: {'source_value': list(cui2names['cui1.1.1'])[0], 'cui': 'cui1.1.1'},
    }}
    res_w_all = {'entities': dict([(nr, d['entities'][1]) for nr, d in enumerate([
        res_w_cui1, res_w_cui11, res_w_cui111])])}

    def test_gets_incorrect_span_big(self, cui='cui1', name='shallow'):
        fr = FailDescriptor.get_reason_for(
            cui, name, self.res_w_cui11, self.tl)
        self.assertIs(fr.reason, FailReason.INCORRECT_SPAN_BIG)

    def test_gets_incorrect_span_bigger(self, cui='cui1', name='shallow'):
        fr = FailDescriptor.get_reason_for(
            cui, name, self.res_w_cui111, self.tl)
        self.assertIs(fr.reason, FailReason.INCORRECT_SPAN_BIG)

    def test_gets_incorrect_span_small(self, cui='cui1.1', name='broader shallow'):
        fr = FailDescriptor.get_reason_for(cui, name, self.res_w_cui1, self.tl)
        self.assertIs(fr.reason, FailReason.INCORRECT_SPAN_SMALL)  # HERE

    def test_gets_incorrect_span_smaller(self, cui='cui1.1.1', name='even broader shallow'):
        fr = FailDescriptor.get_reason_for(cui, name, self.res_w_cui1, self.tl)
        self.assertIs(fr.reason, FailReason.INCORRECT_SPAN_SMALL)  # and HERE

    def test_gets_not_annotated(self, cui='cui2', name='name-2'):
        fr = FailDescriptor.get_reason_for(cui, name, self.res_w_all, self.tl)
        self.assertIs(fr.reason, FailReason.CONCEPT_NOT_ANNOTATED)


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
        ((10, 15, 0, 11), Finding.SPAN_OVERLAP),
        ((10, 15, 0, 15), Finding.BIGGER_SPAN_LEFT),
        ((10, 15, 0, 25), Finding.BIGGER_SPAN_BOTH),
        # start == exp_start
        ((10, 15, 10, 12), Finding.SMALLER_SPAN),
        ((10, 15, 10, 15), Finding.IDENTICAL),
        ((10, 15, 10, 25), Finding.BIGGER_SPAN_RIGHT),
        # exp_start < start < exp_end
        ((10, 15, 12, 13), Finding.SMALLER_SPAN),
        ((10, 15, 12, 15), Finding.SMALLER_SPAN),
        ((10, 15, 12, 25), Finding.SPAN_OVERLAP),
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
         }, Finding.SPAN_OVERLAP),
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=12, end=122)},
         }, Finding.SPAN_OVERLAP),
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
