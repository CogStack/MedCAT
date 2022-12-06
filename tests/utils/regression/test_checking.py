
import unittest

from medcat.utils.regression.targeting import FilterType, FilterStrategy, FilterOptions
from medcat.utils.regression.targeting import TypedFilter, TranslationLayer
from medcat.utils.regression.checking import RegressionChecker, RegressionCase

DICT_WITH_CUI = {'cui': '123'}
DICT_WITH_MULTI_CUI = {'cui': ['111', '101']}
DICT_WITH_NAME = {'name': 'a name'}
DICT_WITH_MULTI_NAME = {'name': ['one name', 'two name']}
DICT_WITH_TYPE_ID = {'type_id': '443'}
DICT_WITH_MULTI_TYPE_ID = {'type_id': ['987', '789']}
# from python 3.6 the following _should_ remember the order of the dict items
# which should mean that the orders in the tests are correct
DICT_WITH_MIX_1 = dict(DICT_WITH_CUI, **DICT_WITH_NAME)
DICT_WITH_MIX_2 = dict(DICT_WITH_NAME, **DICT_WITH_MULTI_TYPE_ID)
DICT_WITH_MIX_3 = dict(DICT_WITH_MULTI_NAME, **DICT_WITH_MULTI_TYPE_ID)
DICT_WITH_MIX_4 = dict(DICT_WITH_MIX_3, **DICT_WITH_MULTI_CUI)


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


class FakeCat:

    def __init__(self, tl: TranslationLayer) -> None:
        self.tl = tl

    def get_entities(self, text, only_cui=True) -> dict:
        if text in self.tl.name2cuis:
            cuis = list(self.tl.name2cuis[text])
            if only_cui:
                return {'entities': dict((i, cui) for i, cui in enumerate(cuis))}
            return {'entities': dict((i, {'cui': cui, 'source_value': text}) for i, cui in enumerate(cuis))}
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
        targets = list(tl.all_targets([ei[0] for ei in EXAMPLE_INFOS], [
                       ei[1] for ei in EXAMPLE_INFOS], [ei[2] for ei in EXAMPLE_INFOS]))
        self.assertEqual(len(targets), len(EXAMPLE_INFOS))


_CUI = 'C123'
_NAME = 'NAMEof123'
_TYPE_ID = '-1'
_D = {'cui': _CUI}
_tts = TypedFilter.from_dict(_D)
_cui2names = {_CUI: [_NAME, ]}
_name2cuis = {_NAME: [_CUI, ]}
_cui2type_ids = {_CUI: [_TYPE_ID, ]}
_cui2children = {}
_tl = TranslationLayer(cui2names=_cui2names, name2cuis=_name2cuis,
                       cui2type_ids=_cui2type_ids, cui2children=_cui2children)


class TestTypedFilter(unittest.TestCase):

    def test_has_correct_target_type(self):
        target_types = [FilterType.CUI, FilterType.NAME, FilterType.TYPE_ID]
        for target_type in target_types:
            with self.subTest(f'With target type {target_type}'):
                tt = TypedFilter(type=target_type, values=[])
                self.assertEqual(tt.type, target_type)

    def check_is_correct_target(self, in_dict: dict, *types, test_with_upper_case=True):
        tts = TypedFilter.from_dict(in_dict)
        # should have the correct number of elements
        self.assertEqual(len(tts), len(types))
        for (the_type, single_multi), tt in zip(types, tts):
            with self.subTest(f'With type {the_type} and {single_multi}'):
                self.assertIsInstance(tt, single_multi)
                self.assertEqual(tt.type, the_type)
        if test_with_upper_case:  # also test upper case
            upper_case_dict = dict((key.upper(), val)
                                   for key, val in in_dict.items())
            self.check_is_correct_target(
                upper_case_dict, *types, test_with_upper_case=False)

    def test_constructs_SingleTarget_from_dict_with_single_cui(self):
        self.check_is_correct_target(
            DICT_WITH_CUI, (FilterType.CUI, TypedFilter))

    def test_constructs_MultiTarget_from_dict_with_multiple_cuis(self):
        self.check_is_correct_target(
            DICT_WITH_MULTI_CUI, (FilterType.CUI, TypedFilter))

    def test_constructs_SingleTarget_from_dict_with_single_name(self):
        self.check_is_correct_target(
            DICT_WITH_NAME, (FilterType.NAME, TypedFilter))

    def test_constructs_MultiTarget_from_dict_with_multiple_names(self):
        self.check_is_correct_target(
            DICT_WITH_MULTI_NAME, (FilterType.NAME, TypedFilter))

    def test_constructs_SingleTarget_from_dict_with_single_type_id(self):
        self.check_is_correct_target(
            DICT_WITH_TYPE_ID, (FilterType.TYPE_ID, TypedFilter))

    def test_constructs_MultiTarget_from_dict_with_multiple_type_ids(self):
        self.check_is_correct_target(
            DICT_WITH_MULTI_TYPE_ID, (FilterType.TYPE_ID, TypedFilter))

    def test_constructs_correct_list_of_types_1(self):
        self.check_is_correct_target(DICT_WITH_MIX_1, (
            FilterType.CUI, TypedFilter), (FilterType.NAME, TypedFilter))

    def test_constructs_correct_list_of_types_2(self):
        self.check_is_correct_target(DICT_WITH_MIX_2, (
            FilterType.NAME, TypedFilter), (FilterType.TYPE_ID, TypedFilter))

    def test_constructs_correct_list_of_types_3(self):
        self.check_is_correct_target(DICT_WITH_MIX_3, (
            FilterType.NAME, TypedFilter), (FilterType.TYPE_ID, TypedFilter))

    def test_constructs_correct_list_of_types_4(self):
        self.check_is_correct_target(DICT_WITH_MIX_4, (
            FilterType.NAME, TypedFilter), (FilterType.TYPE_ID, TypedFilter), (FilterType.CUI, TypedFilter))

    def test_get_applicable_targets_gets_target(self):
        self.assertEqual(len(_tts), 1)
        tt = _tts[0]
        targets = list(tt.get_applicable_targets(_tl, _tl.all_targets([ei[0] for ei in EXAMPLE_INFOS], [
                       ei[1] for ei in EXAMPLE_INFOS], [ei[2] for ei in EXAMPLE_INFOS])))
        self.assertEqual(len(targets), 1)
        cui, name = targets[0]
        self.assertEqual(name, _NAME)
        self.assertEqual(cui, _CUI)

    def test_get_applicable_targets_gets_target_from_many(self):
        # add noise to existing translations
        cui2names = dict(
            _cui2names, **dict((f'{cui}rnd', f'{name}sss') for cui, name in _cui2names.items()))
        name2cuis = dict(
            _name2cuis, **dict((f'{name}sss', f'{cui}123') for cui, name in _name2cuis.items()))
        cui2type_ids = dict(
            _cui2type_ids, **dict((f'{cui}123', 'typeid') for cui in _cui2type_ids))
        cui2children = {}
        tl = TranslationLayer(cui2names=cui2names, name2cuis=name2cuis,
                              cui2type_ids=cui2type_ids, cui2children=cui2children)
        self.assertEqual(len(_tts), 1)
        tt = _tts[0]
        targets = list(tt.get_applicable_targets(tl, tl.all_targets([ei[0] for ei in EXAMPLE_INFOS], [
                       ei[1] for ei in EXAMPLE_INFOS], [ei[2] for ei in EXAMPLE_INFOS])))
        self.assertEqual(len(targets), 1)
        cui, name = targets[0]
        self.assertEqual(name, _NAME)
        self.assertEqual(cui, _CUI)


class TestFilterOptions(unittest.TestCase):

    def test_loads_from_dict(self):
        D = {'strategy': 'all'}
        opts = FilterOptions.from_dict(D)
        self.assertIsInstance(opts, FilterOptions)
        self.assertEqual(opts.strategy, FilterStrategy.ALL)

    def test_loads_from_dict_defaults_not_pref_only(self):
        D = dict()
        opts = FilterOptions.from_dict(D)
        self.assertIsInstance(opts, FilterOptions)
        self.assertFalse(opts.onlyprefnames)

    def test_loads_from_empty_dict_w_default(self):
        D = dict()
        opts = FilterOptions.from_dict(D)
        self.assertIsInstance(opts, FilterOptions)
        self.assertEqual(opts.strategy, FilterStrategy.ALL)

    def test_loads_from_dict_with_onlypref(self):
        D = {'prefname-only': 'True'}
        opts = FilterOptions.from_dict(D)
        self.assertIsInstance(opts, FilterOptions)
        self.assertTrue(opts.onlyprefnames)


class TestRegressionCase(unittest.TestCase):
    D_MIN = {'targeting': {'filters': DICT_WITH_CUI},
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
        self.assertEqual(len(rc.filters), 1)
        self.assertIsInstance(rc.options, FilterOptions)
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
        D['targeting'].pop('filters')
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

    D_COMPLEX = {'targeting': dict({'filters': DICT_WITH_MIX_4}, **{'strategy': 'any',
                                   'prefname-only': 'true'}), 'phrases': ['The phrase %s works', 'ALL %s phrases']}

    def test_loads_from_complex_dict(self):
        NAME = 'NAMEC'
        D = self.complex_d
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        self.assertIsInstance(rc, RegressionCase)
        self.assertEqual(len(rc.filters), 3)
        self.assertIsInstance(rc.options, FilterOptions)
        self.assertEqual(len(rc.phrases), 2)

    TARGET_CUI = 'C123'
    D_SPECIFIC_CASE = {'targeting': {'filters': {
        'cui': [TARGET_CUI, ]}}, 'phrases': ['%s']}  # should just find the name itself

    def test_specific_case_CUI(self):
        NAME = 'NAMESC'
        tl = TranslationLayer.from_CDB(FakeCDB(*EXAMPLE_INFOS))
        D = TestRegressionCase.D_SPECIFIC_CASE
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        success, fail = rc.check_case(FakeCat(tl), tl)
        self.assertEqual(fail, 0)
        self.assertEqual(success, len(
            tl.cui2names[TestRegressionCase.TARGET_CUI]))

    TARGET_NAME = 'N223'
    D_SPECIFIC_CASE_NAME = {'targeting': {'filters': {
        'name': TARGET_NAME}}, 'phrases': ['%s']}

    def test_specific_case_NAME(self):
        NAME = 'NAMESC2'
        tl = TranslationLayer.from_CDB(FakeCDB(*EXAMPLE_INFOS))
        D = TestRegressionCase.D_SPECIFIC_CASE_NAME
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        success, fail = rc.check_case(FakeCat(tl), tl)
        self.assertEqual(fail, 0)
        self.assertEqual(success, len(
            tl.name2cuis[TestRegressionCase.TARGET_NAME]))

    TARGET_TYPE = 'T1'
    D_SPECIFIC_CASE_TYPE_ID = {'targeting': {'filters': {
        'type_id': TARGET_TYPE}}, 'phrases': ['%s']}

    def test_specific_case_TYPE_ID(self):
        NAME = 'NAMESC3'
        tl = TranslationLayer.from_CDB(FakeCDB(*EXAMPLE_INFOS))
        D = TestRegressionCase.D_SPECIFIC_CASE_TYPE_ID
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        success, fail = rc.check_case(FakeCat(tl), tl)
        self.assertEqual(fail, 0)
        self.assertEqual(success, len(EXAMPLE_TYPE_T1_CUI))

    PARENT_CUI = 'C123'
    CHILD_CUI = 'C124'
    D_PARENT_W_CHILDREN = {'targeting': {'filters': {
        'cui_and_children': {'cui': PARENT_CUI, 'depth': 1}}},
        'phrases': ['%s']}
    PT2CHILD = {PARENT_CUI: set([CHILD_CUI])}

    def test_cui_and_children_finds_child(self):
        NAME = 'NAMEpt2ch'
        cdb = FakeCDB(*EXAMPLE_INFOS)
        cdb.addl_info['pt2ch'].update(self.PT2CHILD)
        tl = TranslationLayer.from_CDB(cdb)
        D = self.D_PARENT_W_CHILDREN
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        success, fail = rc.check_case(FakeCat(tl), tl)
        self.assertEqual(fail, 0)
        expected = len(cdb.cui2names[self.PARENT_CUI]) + \
            len(cdb.cui2names[self.CHILD_CUI])
        self.assertEqual(success, expected)

    P_CUI = 'C123'
    C_CUI1 = 'C124'
    C_CUI2 = 'C223'
    C_CUI1_C1 = 'C224'
    C_CUI1_C1_C1 = 'C323'
    C_CUI1_C1_C1_C1 = 'C324'
    D_MULIT_CHILD_1 = {'targeting': {'filters': {
        'cui_and_children': {'cui': P_CUI, 'depth': 2}}},
        'phrases': ['%s']}
    PT2CHILD_M1 = {P_CUI: set([C_CUI1, C_CUI2]),
                   C_CUI1: set([C_CUI1_C1]),
                   C_CUI1_C1: set([C_CUI1_C1_C1]),
                   C_CUI1_C1_C1: set([C_CUI1_C1_C1_C1])}

    def test_cui_and_children_finds_children_depth_2(self):
        NAME = 'NAMEpt2ch'
        cdb = FakeCDB(*EXAMPLE_INFOS)
        cdb.addl_info['pt2ch'].update(self.PT2CHILD_M1)
        tl = TranslationLayer.from_CDB(cdb)
        D = self.D_MULIT_CHILD_1
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        success, fail = rc.check_case(FakeCat(tl), tl)
        self.assertEqual(fail, 0)
        expected = len(cdb.cui2names[self.P_CUI])
        # children
        for child in tl.cui2children[self.P_CUI]:
            expected += len(cdb.cui2names[child])
            # children of children
            for child2 in tl.cui2children[child]:
                expected += len(cdb.cui2names[child2])
        self.assertEqual(success, expected)

    D_MULIT_CHILD_2 = {'targeting': {'filters': {
        'cui_and_children': {'cui': P_CUI, 'depth': 3}}},
        'phrases': ['%s']}

    def test_cui_and_children_finds_children_depth_3(self):
        NAME = 'NAMEpt2ch'
        cdb = FakeCDB(*EXAMPLE_INFOS)
        cdb.addl_info['pt2ch'].update(self.PT2CHILD_M1)
        tl = TranslationLayer.from_CDB(cdb)
        D = self.D_MULIT_CHILD_2
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        success, fail = rc.check_case(FakeCat(tl), tl)
        self.assertEqual(fail, 0)
        expected = len(cdb.cui2names[self.P_CUI])
        # children
        for child in tl.cui2children[self.P_CUI]:
            expected += len(cdb.cui2names[child])
            # children of children
            for child2 in tl.cui2children[child]:
                expected += len(cdb.cui2names[child2])
                # children of children of children
                for child3 in tl.cui2children[child2]:
                    expected += len(cdb.cui2names[child3])
        self.assertEqual(success, expected)

    def test_gets_with_ANY_strategy(self):
        NAME = 'ANYNAME'
        tl = TranslationLayer.from_CDB(FakeCDB(*EXAMPLE_INFOS))
        D = {'targeting': {'strategy': 'any', 'filters': {
            'cui': ['C123', 'C124'], 'name': ['N223', 'N224']}}, 'phrases': ['%s']}
        rc: RegressionCase = RegressionCase.from_dict(NAME, D)
        success, fail = rc.check_case(FakeCat(tl), tl)
        self.assertEqual(fail, 0)
        expected = sum([len(tl.cui2children[cui]) for cui in D['targeting']
                       ['filters']['cui']]) + len(D['targeting']['filters']['name'])
        self.assertEqual(success, expected)


class TestRegressionChecker(unittest.TestCase):

    def test_reads_default(self, yaml_file='configs/default_regression_tests.yml'):
        rc = RegressionChecker.from_yaml(yaml_file)
        self.assertIsInstance(rc, RegressionChecker)
