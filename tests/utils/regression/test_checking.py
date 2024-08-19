import os
import json
import unittest

from medcat.config import Config
from medcat.utils.regression.targeting import OptionSet, FinalTarget
from medcat.utils.regression.targeting import TranslationLayer
from medcat.utils.regression.checking import RegressionSuite, RegressionCase, MetaData
from medcat.utils.regression.results import Finding

EXAMPLE_CUI = '123'
COMPLEX_PLACEHOLDERS = [
    {'placeholder': "[PH1]",
     'cuis': ['cui1', 'cui2']},
    {'placeholder': "[PH2]",
     'cuis': ['cui3', 'cui4']},
    {'placeholder': "[PH3]",
     'cuis': ['cui1', 'cui3']},
]


EXAMPLE_INFOS = [
    # CUI, NAME, TYPE_ID
    ['C123', 'N123', 'T1'],
    ['C124', 'N124', 'T1'],
    ['C223', 'N223', 'T2'],
    ['C224', 'N224', 'T2'],
    # non-unique name
    ['C323', 'N123', 'T3'],
    ['C324', 'N124', 'T3'],
]

EXAMPLE_TYPE_T1_CUI = [cui for cui, _,
                       type_id in EXAMPLE_INFOS if type_id == 'T1']
EXAMPLE_TYPE_T2_CUI = [cui for cui, _,
                       type_id in EXAMPLE_INFOS if type_id == 'T2']
EXAMPLE_TYPE_T3_CUI = [cui for cui, _,
                       type_id in EXAMPLE_INFOS if type_id == 'T3']


class FakeCDB:

    def __init__(self, *infos) -> None:
        self.cui2names = {}
        self.name2cuis = {}
        self.cui2type_ids = {}
        pt2ch = {}
        self.addl_info = {'pt2ch': pt2ch}
        for cui, name, type_id in infos:
            if cui in self.cui2names:
                self.cui2names[cui].add(name)
            else:
                self.cui2names[cui] = set([name])
            if cui in self.cui2type_ids:
                self.cui2type_ids[cui].add(type_id)
            else:
                self.cui2type_ids[cui] = set([type_id])
            if name in self.name2cuis:
                self.name2cuis[name].add(cui)
            else:
                self.name2cuis[name] = set([cui])
        pt2ch.update(dict((cui, set()) for cui in self.cui2names))
        self.cui2preferred_name = {c_cui: list(names)[0] for c_cui, names in self.cui2names.items()}
        self.config = Config()


class FakeCat:

    def __init__(self, tl: TranslationLayer) -> None:
        self.tl = tl

    def get_entities(self, text, only_cui=True) -> dict:
        if text in self.tl.name2cuis:
            cuis = list(self.tl.name2cuis[text])
            if only_cui:
                return {'entities': dict((i, cui) for i, cui in enumerate(cuis))}
            return {'entities': dict((i, {'cui': cui, 'source_value': text, 'start': 0, 'end': 4})
                                     for i, cui in enumerate(cuis))}
        return {}


class TestTranslationLayer(unittest.TestCase):

    def test_TranslationLayer_works_from_empty_fake_CDB(self):
        fakeCDB = FakeCDB()
        tl = TranslationLayer.from_CDB(fakeCDB)
        self.assertIsInstance(tl, TranslationLayer)

    def test_TranslationLayer_works_from_non_empty_fake_CDB(self):
        fakeCDB = FakeCDB(*EXAMPLE_INFOS)
        tl = TranslationLayer.from_CDB(fakeCDB)
        self.assertIsInstance(tl, TranslationLayer)

    def test_gets_all_targets(self):
        fakeCDB = FakeCDB(*EXAMPLE_INFOS)
        tl = TranslationLayer.from_CDB(fakeCDB)
        targets = [name for ei in EXAMPLE_INFOS for name in tl.get_names_of(ei[0], False)]
        self.assertEqual(len(targets), len(EXAMPLE_INFOS))


class TestRegressionCase(unittest.TestCase):
    D_MIN = {'targeting': {
                'placeholders': [
                    {
                        'placeholder': '%s',
                        'cuis': [EXAMPLE_CUI],
                    }]},
             'phrases': ['The phrase %s works']}

    def _create_copy(self, d):
        import copy
        return copy.deepcopy(d)

    @property
    def min_d(self):  # create copy
        return self._create_copy(TestRegressionCase.D_MIN)

    @property
    def complex_d(self):  # create copy
        return self._create_copy(TestRegressionCase.D_COMPLEX)

    def test_loads_from_min_dict(self):
        NAME = 'NAME0'
        D = self.min_d
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        self.assertIsInstance(rc, RegressionCase)
        self.assertEqual(len(rc.options.options), 1)
        self.assertIsInstance(rc.options, OptionSet)
        self.assertEqual(len(rc.phrases), 1)

    def test_fails_dict_no_targets_1(self):
        NAME = 'NAME1'
        D = self.min_d
        D.pop('targeting')
        with self.assertRaises(ValueError):
            RegressionCase.from_dict(NAME, D)

    def test_fails_dict_no_targets_2(self):
        NAME = 'NAME2'
        D = self.min_d
        D['targeting'].pop('placeholders')
        with self.assertRaises(ValueError):
            RegressionCase.from_dict(NAME, D)

    def test_fails_with_no_phrases_1(self):
        NAME = 'NAMEF'
        D = self.min_d
        D['phrases'] = []
        with self.assertRaises(ValueError):
            RegressionCase.from_dict(NAME, D)

    def test_fails_with_no_phrases_2(self):
        NAME = 'NAMEF'
        D = self.min_d
        D.pop('phrases')
        with self.assertRaises(ValueError):
            RegressionCase.from_dict(NAME, D)

    D_COMPLEX = {'targeting': {'placeholders': COMPLEX_PLACEHOLDERS},
                 'phrases': ['The phrase %s works', 'ALL %s phrases']}

    def test_loads_from_complex_dict(self):
        NAME = 'NAMEC'
        D = self.complex_d
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        self.assertIsInstance(rc, RegressionCase)
        self.assertEqual(len(rc.options.options), 3)
        self.assertIsInstance(rc.options, OptionSet)
        self.assertEqual(len(rc.phrases), 2)

    TARGET_CUI = 'C123'
    D_SPECIFIC_CASE = {'targeting': {'placeholders': [{
            'placeholder': '%s',
            'cuis': [TARGET_CUI, ]}
        ]}, 'phrases': ['%s']}  # should just find the name itself


class TestRegressionCaseCheckModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        NAME = 'NAMESC'
        cls.tl = TranslationLayer.from_CDB(FakeCDB(*EXAMPLE_INFOS))
        D = TestRegressionCase.D_SPECIFIC_CASE
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        regr_checker = RegressionSuite([rc], MetaData.unknown())
        cls.res = regr_checker.check_model(FakeCat(cls.tl), cls.tl)

    def test_specific_case_CUI(self):
        findings = self.res.findings
        fail = findings.get(Finding.FAIL, 0)
        success = sum(v for f, v in findings.items() if f is not Finding.FAIL)
        self.assertEqual(fail, 0)
        self.assertEqual(success, len(
            self.tl.cui2names[TestRegressionCase.TARGET_CUI]))


class TestRegressionCaseCheckModelJson(TestRegressionCaseCheckModel):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # add a non-perfect example to show in the below
        cls.res.parts[0].examples.append((FinalTarget(placeholder='PH', cui='CUI_PARENT',
                                                      name='NAME_PARENT',
                                                      final_phrase="FINAL PHRASE"),
                                          (Finding.FOUND_ANY_CHILD, 'CHILD')))

    def test_result_is_json_serialisable(self):
        rd = self.res.dict()
        s = json.dumps(rd)
        self.assertIsInstance(s, str)

    def test_result_is_json_serialisable_pydantic(self):
        s = self.res.json()
        self.assertIsInstance(s, str)

    def test_can_use_strictness(self):
        e1 = [
            example for part in self.res.dict(strictness='STRICTEST')['parts']
            for example in part['examples']
        ]
        e2 = [
            example for part in self.res.dict(strictness='LENIENT')['parts']
            for example in part['examples']
        ]
        self.assertGreater(len(e1), len(e2))


class TestRegressionChecker(unittest.TestCase):
    YAML_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                             "configs", "default_regression_tests.yml")
    MCT_EXPORT_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                                   'resources', 'medcat_trainer_export.json')

    @classmethod
    def setUpClass(cls) -> None:
        cls.rc = RegressionSuite.from_yaml(cls.YAML_PATH)

    def test_reads_correctly(self):
        self.assertIsInstance(self.rc, RegressionSuite)

    def test_has_cases(self):
        self.assertGreater(len(self.rc.cases), 0)


class TestRegressionCheckerFromMCTExport(TestRegressionChecker):

    @classmethod
    def setUpClass(cls) -> None:
        cls.rc = RegressionSuite.from_mct_export(cls.MCT_EXPORT_PATH)


class MultiPlaceholderTests(unittest.TestCase):
    THE_DICT = {
        "mulit-placeholder-case": {
            'targeting': {
                'placeholders': [
                    {
                        'placeholder': '[CONCEPT]',
                        'cuis': ['C123', 'C124']
                        # either has 1 name
                    }
                ]
            },
            'phrases': [
                "This [CONCEPT] has mulitple [CONCEPT] instances of [CONCEPT]"
                # 3 instances
            ]
        }
    }
    EXPECTED_CASES = 2 * 1 * 3  # 2 CUIs, 1 name each, 3 placeholders
    FAKE_CDB = FakeCDB(*EXAMPLE_INFOS)
    TL = TranslationLayer.from_CDB(FAKE_CDB)

    @classmethod
    def setUpClass(cls) -> None:
        cls.rc = RegressionSuite.from_dict(cls.THE_DICT)

    def test_reads_successfully(self):
        self.assertIsInstance(self.rc, RegressionSuite)

    def test_gets_cases(self):
        cases = list(self.rc.iter_subcases(self.TL))
        self.assertEqual(len(cases), self.EXPECTED_CASES)
