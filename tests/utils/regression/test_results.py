
from typing import Optional
import unittest
from copy import deepcopy
import json

from medcat.utils.regression.targeting import TranslationLayer
from medcat.utils.regression.results import Finding, MalformedFinding
from medcat.utils.regression.results import FindingDeterminer
from medcat.utils.regression.results import SingleResultDescriptor
from medcat.utils.regression.targeting import FinalTarget

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
        # start from example 12
        # FAILURES
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(cui="CUI2")},
         }, Finding.FOUND_OTHER),
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=0, end=5)},
         }, Finding.FAIL),
        ({**_get_example_kwargs(),
          "found_entities": {0: _get_example_ent(start=20, end=25)},
         }, Finding.FAIL),
        ({**_get_example_kwargs(),
          "found_entities": {},
         }, Finding.FAIL),
    ]
    NR_OF_EXAMPLES = len(EXAMPLES)
    TL = TranslationLayer.from_CDB(FakeCDB())

    def test_finds_examples(self):
        self.assertEqual(len(self.EXAMPLES), self.NR_OF_EXAMPLES)
        for nr, (ekwargs, expected) in enumerate(self.EXAMPLES):
            with self.subTest(f"With [{nr}] kwargs {ekwargs}"):
                found, _ = Finding.determine(tl=self.TL, **ekwargs)
                self.assertEqual(found, expected)


EXAMPLE_INFOS = [
    ['CGP', 'NGP', 'T1'],  # the grandparent
    # CUI, NAME, TYPE_ID
    ['C123', 'N123', 'T1'],
    ['C124', 'N124', 'T1'],
    ['C223', 'N223', 'T2'],
    ['C224', 'N224', 'T2'],
    # non-unique name
    ['C323', 'N123', 'T3'],
    ['C324', 'N124', 'T3'],
]


class FindingFromEntsWithChildrenTests(unittest.TestCase):
    FAKE_CDB = FakeCDB(*EXAMPLE_INFOS)
    TL = TranslationLayer.from_CDB(FAKE_CDB)
    THE_GRANPARENT = 'CGP'
    THE_PARENT = "C123"
    THE_CHILD = "C124"
    PT2CHILD = {
        THE_GRANPARENT: {THE_PARENT},
        THE_PARENT: {THE_CHILD}
    }
    CHILD_MAPPED_EXACT_SPAN = {**_get_example_kwargs(cui=THE_PARENT),
                               "found_entities": {0: _get_example_ent(cui=THE_CHILD)}}
    GRANDCHILD_MAPPED_EXACT_SPAN = {
        **_get_example_kwargs(cui=THE_GRANPARENT),
        "found_entities": {0: _get_example_ent(cui=THE_CHILD)}}
    CHILD_MAPPED_PARTIAL_SAPN1 = {**_get_example_kwargs(cui=THE_PARENT),
                                  "found_entities": {0: _get_example_ent(cui=THE_CHILD, start=5, end=14)}}
    CHILD_MAPPED_PARTIAL_SAPN2 = {**_get_example_kwargs(cui=THE_PARENT),
                                  "found_entities": {0: _get_example_ent(cui=THE_CHILD, start=5, end=15)}}
    CHILD_MAPPED_PARTIAL_SAPN3 = {**_get_example_kwargs(cui=THE_PARENT),
                                  "found_entities": {0: _get_example_ent(cui=THE_CHILD, start=5, end=20)}}
    CHILD_MAPPED_PARTIAL_SAPN4 = {**_get_example_kwargs(cui=THE_PARENT),
                                  "found_entities": {0: _get_example_ent(cui=THE_CHILD, start=10, end=14)}}
    CHILD_MAPPED_PARTIAL_SAPN5 = {**_get_example_kwargs(cui=THE_PARENT),
                                  "found_entities": {0: _get_example_ent(cui=THE_CHILD, start=10, end=20)}}
    CHILD_MAPPED_PARTIAL_SAPN6 = {**_get_example_kwargs(cui=THE_PARENT),
                                  "found_entities": {0: _get_example_ent(cui=THE_CHILD, start=11, end=14)}}
    CHILD_MAPPED_PARTIAL_SAPN7 = {**_get_example_kwargs(cui=THE_PARENT),
                                  "found_entities": {0: _get_example_ent(cui=THE_CHILD, start=11, end=15)}}
    CHILD_MAPPED_PARTIAL_SAPN8 = {**_get_example_kwargs(cui=THE_PARENT),
                                  "found_entities": {0: _get_example_ent(cui=THE_CHILD, start=11, end=20)}}
    PARTIAL_CHILDREN = [
        CHILD_MAPPED_PARTIAL_SAPN1, CHILD_MAPPED_PARTIAL_SAPN2, CHILD_MAPPED_PARTIAL_SAPN3,
        CHILD_MAPPED_PARTIAL_SAPN4, CHILD_MAPPED_PARTIAL_SAPN5, CHILD_MAPPED_PARTIAL_SAPN6,
        CHILD_MAPPED_PARTIAL_SAPN7, CHILD_MAPPED_PARTIAL_SAPN8
    ]
    PARTIAL_GRANDCHILDREN = [
        {**d, "exp_cui": 'CGP'} for d in deepcopy(PARTIAL_CHILDREN)]
    PARENT_MAPPED_EXACT_SPAN = {
        **_get_example_kwargs(cui=THE_CHILD),
        "found_entities": {0: _get_example_ent(cui=THE_PARENT)}
    }
    GRANDPARENT_MAPPED_EXACT_SPAN = {
        **_get_example_kwargs(cui=THE_CHILD),
        "found_entities": {0: _get_example_ent(cui=THE_GRANPARENT)}
    }

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.FAKE_CDB.addl_info['pt2ch'].update(cls.PT2CHILD)

    def test_finds_child_exact_span(self):
        finding, optcui = Finding.determine(tl=self.TL, **self.CHILD_MAPPED_EXACT_SPAN)
        self.assertIs(finding, Finding.FOUND_ANY_CHILD)
        self.assertIsNotNone(optcui)
        self.assertTrue(optcui.startswith(self.THE_CHILD))

    def test_finds_grandchild_exact_span(self):
        finding, optcui = Finding.determine(tl=self.TL, **self.GRANDCHILD_MAPPED_EXACT_SPAN)
        self.assertIs(finding, Finding.FOUND_ANY_CHILD)
        self.assertIsNotNone(optcui)
        self.assertTrue(optcui.startswith(self.THE_CHILD))

    def test_finds_child_partial_span(self):
        for nr, ekwargs in enumerate(self.PARTIAL_CHILDREN):
            with self.subTest(f"{nr}: {ekwargs}"):
                finding, optcui = Finding.determine(tl=self.TL, **ekwargs)
                self.assertIs(finding, Finding.FOUND_CHILD_PARTIAL)
                self.assertIsNotNone(optcui)
                self.assertTrue(optcui.startswith(self.THE_CHILD))

    def test_finds_grandchild_partial_span(self):
        for nr, ekwargs in enumerate(self.PARTIAL_GRANDCHILDREN):
            with self.subTest(f"{nr}: {ekwargs}"):
                finding, optcui = Finding.determine(tl=self.TL, **ekwargs)
                self.assertIs(finding, Finding.FOUND_CHILD_PARTIAL)
                self.assertIsNotNone(optcui)
                self.assertTrue(optcui.startswith(self.THE_CHILD))

    def test_finds_parent_exact_span(self):
        finding, parcui = Finding.determine(tl=self.TL, **self.PARENT_MAPPED_EXACT_SPAN)
        self.assertIs(finding, Finding.FOUND_DIR_PARENT)
        self.assertTrue(parcui.startswith(self.THE_PARENT))  # NOTE: also has the preferred name

    def test_finds_grandparent_exact_span(self):
        finding, parcui = Finding.determine(tl=self.TL, **self.GRANDPARENT_MAPPED_EXACT_SPAN)
        self.assertIs(finding, Finding.FOUND_DIR_GRANDPARENT)
        self.assertTrue(parcui.startswith(self.THE_GRANPARENT))  # NOTE: also has the preferred name


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
                found, optcui = Finding.determine(tl=self.TL, **ekwargs)
                self.assertEqual(found, Finding.FAIL)
                self.assertIsNotNone(optcui)


class SingleResultDescriptorSerialisationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        e1 = (FinalTarget(placeholder='$', cui='CUI1', name='NAME1', final_phrase='FINAL PHRASE'),
              (Finding.FOUND_OTHER, 'OTHER CUI'))
        e2 = (FinalTarget(placeholder='$', cui='CUIP', name='PARENT', final_phrase='FINAL PHRASE'),
              (Finding.FOUND_ANY_CHILD, 'CUI_C (CHILD)'))
        findings = {Finding.FOUND_OTHER: 1, Finding.FOUND_ANY_CHILD: 1}
        cls.rd = SingleResultDescriptor(name="RANDOM_NAME", findings=findings,
                                        examples=[e1, e2])

    def test_can_json_dump_pydantic(self):
        s = self.rd.json()
        self.assertIsInstance(s, str)

    def test_can_json_dump_json(self):
        s = json.dumps(self.rd.model_dump())
        self.assertIsInstance(s, str)

    def test_can_use_strictness_for_dump(self):
        d_strictest = self.rd.model_dump(strictness='STRICTEST')
        e_strictest = d_strictest['examples']
        # this should have more examples
        d_lenient = self.rd.model_dump(strictness='NORMAL')
        e_normal = d_lenient['examples']
        self.assertGreater(len(e_strictest), len(e_normal))
