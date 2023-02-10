import os
from typing import Iterator, cast
import yaml
from functools import lru_cache
import tempfile

from medcat.utils.regression.checking import RegressionCase, ResultDescriptor, FilterOptions, FilterStrategy, TypedFilter, FilterType
from medcat.utils.regression.checking import RegressionChecker
from medcat.utils.regression.converting import medcat_export_json_to_regression_yml
from medcat.utils.regression.category_separation import CategoryDescription, Category, AllPartsCategory, AnyPartOfCategory
from medcat.utils.regression.category_separation import SeparationObserver, SeparateToFirst, SeparateToAll, read_categories
from medcat.utils.regression.category_separation import RegressionCheckerSeparator, separate_categories, StrategyType
from medcat.utils.regression.editing import combine_yamls

import unittest


class CategoryDescriptionTests(unittest.TestCase):
    CUIS = ['c123', 'c111']
    NAMES = ['NAME1', 'NAME9']
    TUIS = ['T-1', 'T-10']

    def setUp(self) -> None:
        self.cd = CategoryDescription(
            target_cuis=set(self.CUIS), target_names=set(self.NAMES), target_tuis=set(self.TUIS))
        self.anything = CategoryDescription.anything_goes()

    def test_initiates(self):
        self.assertIsNotNone(self.cd)

    def get_case_for(self, cui=None, name=None, tui=None) -> RegressionCase:
        cname = f'TEMPNAME={cui}-{name}-{tui}'
        cphrase = 'does not matter %s'
        fo = FilterOptions(strategy=FilterStrategy.ANY)
        if cui:
            ft = FilterType.CUI
            value = cui
        elif name:
            ft = FilterType.NAME
            value = name
        elif tui:
            ft = FilterType.TYPE_ID
            value = tui
        else:
            raise ValueError(
                f"Unknown filter for CUI: {cui} NAME: {name} and TUI: {tui}")
        cfilter = TypedFilter(type=ft, values=[value])
        return RegressionCase(name=cname, options=fo, filters=[cfilter], phrases=[cphrase], report=ResultDescriptor(name=cname))

    def helper_recognizes(self, items: list, case_kw: str, method: callable):
        for item in items:
            with self.subTest(f'With item {item}, testing {case_kw} and {method} for RECOGNIZES'):
                self.assertTrue(method(self.get_case_for(**{case_kw: item})))

    def helper_does_not_recognize(self, items: list, case_kw: str, method: callable):
        for item in items:
            with self.subTest(f'With item {item}, testing {case_kw} and {method} for NOT RECOGNIZES'):
                self.assertFalse(method(self.get_case_for(**{case_kw: item})))

    def test_recognizes_CUIS(self):
        self.helper_recognizes(self.CUIS, 'cui', self.cd.has_cui_from)

    def test_does_NOT_recognize_wrong_CUIS(self):
        self.helper_does_not_recognize(self.NAMES, 'cui', self.cd.has_cui_from)

    def test_recognizes_NAMES(self):
        self.helper_recognizes(self.NAMES, 'name', self.cd.has_name_from)

    def test_does_NOT_recognize_wrong_NAMES(self):
        self.helper_does_not_recognize(
            self.CUIS, 'name', self.cd.has_name_from)

    def test_recognizes_TUIS(self):
        self.helper_recognizes(self.TUIS, 'tui', self.cd.has_tui_from)

    def test_does_NOT_recognize_wrong_TUIS(self):
        self.helper_does_not_recognize(self.NAMES, 'tui', self.cd.has_tui_from)

    def test_anythong_goes_recognizes_anything_cui4cui(self):
        self.helper_recognizes(self.CUIS, 'cui', self.anything.has_cui_from)

    def test_anythong_goes_recognizes_anything_tui4cui(self):
        self.helper_recognizes(self.TUIS, 'cui', self.anything.has_cui_from)

    def test_anythong_goes_recognizes_anything_name4cui(self):
        self.helper_recognizes(self.NAMES, 'cui', self.anything.has_cui_from)

    def test_anythong_goes_recognizes_anything_tui4tui(self):
        self.helper_recognizes(self.TUIS, 'tui', self.anything.has_tui_from)

    def test_anythong_goes_recognizes_anything_cui4tui(self):
        self.helper_recognizes(self.CUIS, 'tui', self.anything.has_tui_from)

    def test_anythong_goes_recognizes_anything_name4tui(self):
        self.helper_recognizes(self.NAMES, 'tui', self.anything.has_tui_from)

    def test_anythong_goes_recognizes_anything_name4name(self):
        self.helper_recognizes(self.NAMES, 'name', self.anything.has_name_from)

    def test_anythong_goes_recognizes_anything_cui4name(self):
        self.helper_recognizes(self.CUIS, 'name', self.anything.has_name_from)

    def test_anythong_goes_recognizes_anything_tui4name(self):
        self.helper_recognizes(self.TUIS, 'name', self.anything.has_name_from)


def get_case(cui, tui, name):
    if cui:
        cui_filter = TypedFilter(type=FilterType.CUI, values=[cui])
    else:
        cui_filter = None
    if tui:
        tui_filter = TypedFilter(type=FilterType.TYPE_ID, values=[tui])
    else:
        tui_filter = None
    if name:
        name_filter = TypedFilter(type=FilterType.NAME, values=[name])
    else:
        name_filter = None
    fo = FilterOptions(strategy=FilterStrategy.ALL)
    cphrase = 'Phrase does not matter %s'
    filters = [cui_filter, tui_filter, name_filter]
    filters = [f for f in filters if f is not None]
    return RegressionCase(name=f'rc w/ cui: {cui}, tui: {tui}, name: {name}', options=fo,
                          filters=filters, phrases=[cphrase], report=ResultDescriptor(name='TestRD'))


class AllPartsCategoryTests(unittest.TestCase):

    def setUp(self) -> None:
        cdt = CategoryDescriptionTests()
        cdt.setUp()
        self.cat = AllPartsCategory('ALL=parts', cdt.cd)

    def test_initializes(self):
        self.assertIsNotNone(self.cat)

    def test_recognizes_correct(self):
        for cui in CategoryDescriptionTests.CUIS:
            for tui in CategoryDescriptionTests.TUIS:
                for name in CategoryDescriptionTests.NAMES:
                    with self.subTest(f'cui: {cui}, tui: {tui}, name: {name}'):
                        case = get_case(cui, tui, name)
                        self.assertTrue(self.cat.fits(case))

    def helper_does_NOT_recognize_one_at_time_3in1(self, items: list):
        for item in items:
            with self.subTest(f'ITEM: {item} (as CUI, TUI, and name)'):
                case = get_case(item, item, item)
                self.assertFalse(self.cat.fits(case))

    def test_does_NOT_recognize_one_at_time_CUI_3in1(self):
        self.helper_does_NOT_recognize_one_at_time_3in1(
            CategoryDescriptionTests.CUIS)

    def test_does_NOT_recognize_one_at_time_NAME_3in1(self):
        self.helper_does_NOT_recognize_one_at_time_3in1(
            CategoryDescriptionTests.NAMES)

    def test_does_NOT_recognize_one_at_time_TUI_3in1(self):
        self.helper_does_NOT_recognize_one_at_time_3in1(
            CategoryDescriptionTests.TUIS)

    def helper_does_NOT_recognize_one_at_time_just1(self, items: list, order: int):
        args = [None, None, None]
        for item in items:
            with self.subTest(f'ITEM: {item} (as CUI, TUI, OR name)'):
                args[order] = item
                case = get_case(*args)
                self.assertFalse(self.cat.fits(case))

    def test_does_NOT_recognize_one_at_time_CUI_just1(self):
        self.helper_does_NOT_recognize_one_at_time_just1(
            CategoryDescriptionTests.CUIS, 0)

    def test_does_NOT_recognize_one_at_time_CUI_just1_wrong_type1(self):
        self.helper_does_NOT_recognize_one_at_time_just1(
            CategoryDescriptionTests.CUIS, 1)

    def test_does_NOT_recognize_one_at_time_CUI_just1_wrong_type2(self):
        self.helper_does_NOT_recognize_one_at_time_just1(
            CategoryDescriptionTests.CUIS, 2)

    def test_does_NOT_recognize_one_at_time_NAME_just1(self):
        self.helper_does_NOT_recognize_one_at_time_just1(
            CategoryDescriptionTests.NAMES, 2)

    def test_does_NOT_recognize_one_at_time_NAME_just1_wrong_type1(self):
        self.helper_does_NOT_recognize_one_at_time_just1(
            CategoryDescriptionTests.NAMES, 0)

    def test_does_NOT_recognize_one_at_time_NAME_just1_wrong_type2(self):
        self.helper_does_NOT_recognize_one_at_time_just1(
            CategoryDescriptionTests.NAMES, 1)

    def test_does_NOT_recognize_one_at_time_TUI_just1(self):
        self.helper_does_NOT_recognize_one_at_time_just1(
            CategoryDescriptionTests.TUIS, 1)

    def test_does_NOT_recognize_one_at_time_TUI_just1_wrong_type1(self):
        self.helper_does_NOT_recognize_one_at_time_just1(
            CategoryDescriptionTests.TUIS, 0)

    def test_does_NOT_recognize_one_at_time_TUI_just1_wrong_type2(self):
        self.helper_does_NOT_recognize_one_at_time_just1(
            CategoryDescriptionTests.TUIS, 2)


class AnyPartOfCategoryTests(unittest.TestCase):

    def setUp(self) -> None:
        cdt = CategoryDescriptionTests()
        cdt.setUp()
        self.cat = AnyPartOfCategory('ANY=parts', cdt.cd)

    def test_init(self):
        self.assertIsNotNone(self.cat)

    def helper_recognizes_any(self, items: list, order: int):
        args = [None, None, None]
        for item in items:
            with self.subTest(f'Testing {item} as {["CUI", "TUI", "NAME"][order]}'):
                args[order] = item
                case = get_case(*args)
                self.assertTrue(self.cat.fits(case))

    def helper_recognizes_any_2(self, items1: list, order1: int, items2, order2: int):
        args = [None, None, None]
        for item1 in items1:
            for item2 in items2:
                with self.subTest(f'Testing {item1} and {item2} as {["CUI", "TUI", "NAME"][order1]} and '
                                  '{["CUI", "TUI", "NAME"][order2]}, respectively'):
                    args[order1] = item1
                    args[order2] = item2
                    case = get_case(*args)
                    self.assertTrue(self.cat.fits(case))

    def test_recognizes_any_cui(self):
        self.helper_recognizes_any(CategoryDescriptionTests.CUIS, 0)

    def test_recognizes_any_tui(self):
        self.helper_recognizes_any(CategoryDescriptionTests.TUIS, 1)

    def test_recognizes_any_name(self):
        self.helper_recognizes_any(CategoryDescriptionTests.NAMES, 2)

    def test_recognizes_combinations_of_2(self):
        self.helper_recognizes_any_2(
            CategoryDescriptionTests.CUIS, 0, CategoryDescriptionTests.TUIS, 1)
        self.helper_recognizes_any_2(
            CategoryDescriptionTests.CUIS, 0, CategoryDescriptionTests.NAMES, 2)
        self.helper_recognizes_any_2(
            CategoryDescriptionTests.TUIS, 1, CategoryDescriptionTests.NAMES, 2)

    def test_recognizes_combinations_of_3(self):
        for cui in CategoryDescriptionTests.CUIS:
            for tui in CategoryDescriptionTests.TUIS:
                for name in CategoryDescriptionTests.NAMES:
                    with self.subTest(f'cui: {cui}, tui: {tui}, name: {name}'):
                        case = get_case(cui, tui, name)
                        self.assertTrue(self.cat.fits(case))


def get_all_cases() -> Iterator[RegressionCase]:
    for cui in CategoryDescriptionTests.CUIS:
        for tui in CategoryDescriptionTests.TUIS:
            for name in CategoryDescriptionTests.NAMES:
                all_args = [cui, tui, name]
                # unique combinations of 1 or 2 None's
                for nr in range(1, 2**3):  # ignore 0
                    cur_args = [(arg if (nr >> i) & 1 else None)
                                for i, arg in enumerate(all_args)]
                    yield get_case(*cur_args)


class SeparationObserverTests(unittest.TestCase):

    def setUp(self) -> None:
        self.observer = SeparationObserver()
        apct = AnyPartOfCategoryTests()
        apct.setUp()
        self.cat = apct.cat

    def test_init(self):
        self.assertIsNotNone(self.observer)

    def test_new_has_not_observed(self):
        for case in get_all_cases():
            with self.subTest(f'CASE: {case}'):
                self.assertFalse(self.observer.has_observed(case))

    def test_observes(self):
        for case in get_all_cases():
            with self.subTest(f'CASE: {case}'):
                self.observer.observe(case, category=self.cat)
                self.assertTrue(self.observer.has_observed(case))


TEST_CATEGORIES_FILE = os.path.join(
    'tests', 'resources', 'test_categories.yml')


def get_all_categories() -> Iterator[Category]:
    return read_categories(TEST_CATEGORIES_FILE)


class SeparateToFirstTests(unittest.TestCase):

    def setUp(self) -> None:
        sot = SeparationObserverTests()
        sot.setUp()
        self.strat = SeparateToFirst(observer=sot.observer)

    def test_init(self):
        self.assertIsNotNone(self.strat)

    def test_new_can_separates(self):
        for case in get_all_cases():
            with self.subTest(f'CASE: {case}'):
                self.assertTrue(self.strat.can_separate(case))

    def test_separates_cases_with_cat_cui(self):
        for cat in get_all_categories():
            cat = cast(AnyPartOfCategory, cat)
            for cui in cat.description.target_cuis:
                case = get_case(cui, None, None)
                with self.subTest(f'CASE: {case} and CATEGORY {cat}'):
                    self.assertTrue(self.strat.can_separate(case))
                    self.strat.separate(case, cat)
                    self.assertTrue(self.strat.observer.has_observed(case))

    def test_can_not_separate_cases_with_after_initial_separation(self):
        self.test_separates_cases_with_cat_cui()  # do initial separation
        for cat in get_all_categories():
            cat = cast(AnyPartOfCategory, cat)
            for cui in cat.description.target_cuis:
                case = get_case(cui, None, None)
                with self.subTest(f'CASE: {case} and CATEGORY {cat}'):
                    self.assertFalse(self.strat.can_separate(case))


class SeparateToAllTests(unittest.TestCase):

    def setUp(self) -> None:
        sot = SeparationObserverTests()
        sot.setUp()
        self.strat = SeparateToAll(observer=sot.observer)

    def test_init(self):
        self.assertIsNotNone(self.strat)

    def test_new_can_separates(self):
        for case in get_all_cases():
            with self.subTest(f'CASE: {case}'):
                self.assertTrue(self.strat.can_separate(case))

    def test_separates_cases_with_cat_cui(self):
        for cat in get_all_categories():
            cat = cast(AnyPartOfCategory, cat)
            for cui in cat.description.target_cuis:
                case = get_case(cui, None, None)
                with self.subTest(f'CASE: {case} and CATEGORY {cat}'):
                    self.assertTrue(self.strat.can_separate(case))
                    self.strat.separate(case, cat)
                    self.assertTrue(self.strat.observer.has_observed(case))

    def test_can_separate_cases_with_after_initial_separation(self):
        self.test_separates_cases_with_cat_cui()  # do initial separation
        for cat in get_all_categories():
            cat = cast(AnyPartOfCategory, cat)
            for cui in cat.description.target_cuis:
                case = get_case(cui, None, None)
                with self.subTest(f'CASE: {case} and CATEGORY {cat}'):
                    self.assertTrue(self.strat.can_separate(case))


TEST_MCT_EXPORT_JSON_FILE = os.path.join("tests", "resources",
                                         "medcat_trainer_export.json")


@lru_cache
def get_real_checker() -> RegressionChecker:
    yaml_str = medcat_export_json_to_regression_yml(TEST_MCT_EXPORT_JSON_FILE)
    d = yaml.safe_load(yaml_str)
    return RegressionChecker.from_dict(d)


@lru_cache
def get_all_real_cases() -> Iterator[RegressionCase]:
    rc = get_real_checker()
    for case in rc.cases:
        yield case


class RegressionCheckerSeparator_toFirst_Tests(unittest.TestCase):

    def setUp(self) -> None:
        observer = SeparationObserver()
        strat = SeparateToFirst(observer)
        self.separator = RegressionCheckerSeparator(
            categories=list(get_all_categories()), strategy=strat)

    def test_init(self):
        self.assertIsNotNone(self.separator)

    def test_finds_categories(self):
        for case in get_all_real_cases():
            with self.subTest(f'CASE: {case} and {self.separator}'):
                self.separator.find_categories_for(case)
                self.assertTrue(
                    self.separator.strategy.observer.has_observed(case))

    def test_nr_of_cases_remains_same(self):
        nr_of_total_cases = len(list(get_all_real_cases()))
        separated_cases = 0
        self.test_finds_categories()
        for cases in self.separator.strategy.observer.separated.values():
            separated_cases += len(cases)
        self.assertEqual(nr_of_total_cases, separated_cases)


class RegressionCheckerSeparator_toAll_Tests(unittest.TestCase):

    def setUp(self) -> None:
        stat = SeparateToAllTests()
        stat.setUp()
        self.separator = RegressionCheckerSeparator(
            categories=list(get_all_categories()), strategy=stat.strat)

    def test_init(self):
        self.assertIsNotNone(self.separator)

    def test_finds_categories(self):
        for case in get_all_real_cases():
            with self.subTest(f'CASE: {case}'):
                self.separator.find_categories_for(case)
                self.assertTrue(
                    self.separator.strategy.observer.has_observed(case))

    def test_nr_of_cases_remains_same_or_greater(self):
        nr_of_total_cases = len(list(get_all_real_cases()))
        separated_cases = 0
        self.test_finds_categories()
        for cases in self.separator.strategy.observer.separated.values():
            separated_cases += len(cases)
        self.assertGreaterEqual(nr_of_total_cases, separated_cases)


def get_applicable_files_in(folder: str, avoid_basename_start: str = 'converted') -> list:
    orig_list = os.listdir(folder)
    return [os.path.join(folder, fn) for fn in orig_list
            if fn.endswith(".yml") and not fn.startswith(avoid_basename_start)]


class FullSeparationTests(unittest.TestCase):

    def save_copy_with_one_fewer_category(self):
        self.one_fewer_categories_file = os.path.join(
            self.other_temp_folder.name, 'one_fewer_categories.yml')
        with open(TEST_CATEGORIES_FILE) as f:
            d = yaml.safe_load(f)
        categories = d['categories']
        to_remove = list(categories.keys())[0]
        del categories[to_remove]
        yaml_str = yaml.safe_dump(d)
        with open(self.one_fewer_categories_file, 'w') as f:
            f.write(yaml_str)

    def setUp(self) -> None:
        # new temporary folders for new tests, just in case
        self.target_prefix_file = tempfile.TemporaryDirectory()
        self.other_temp_folder = tempfile.TemporaryDirectory()
        self.rc = get_real_checker()
        self.regr_yaml_file = os.path.join(
            self.target_prefix_file.name, "converted_regr.yml")
        yaml_str = self.rc.to_yaml()
        with open(self.regr_yaml_file, 'w') as f:
            f.write(yaml_str)
        self.save_copy_with_one_fewer_category()

    def tearDown(self) -> None:
        self.target_prefix_file.cleanup()
        self.other_temp_folder.cleanup()

    def join_back_up(self) -> RegressionChecker:
        files = get_applicable_files_in(self.target_prefix_file.name)
        f0 = files[0]
        f_new = os.path.join(self.target_prefix_file.name, "join-back-1.yml")
        for f1 in files[1:]:
            combine_yamls(f0, f1, f_new)
            f0 = f_new
        return RegressionChecker.from_yaml(f_new)

    def test_separations_work_alone(self):
        prefix = os.path.join(self.target_prefix_file.name, 'split-')
        separate_categories(TEST_CATEGORIES_FILE,
                            StrategyType.FIRST, self.regr_yaml_file, prefix)
        files = get_applicable_files_in(self.target_prefix_file.name)
        for f in files:
            with self.subTest(f'With {f}'):
                rc = RegressionChecker.from_yaml(f)
                self.assertIsNotNone(rc)

    def test_separations_combined_same(self):
        prefix = os.path.join(self.target_prefix_file.name, 'split-')
        separate_categories(TEST_CATEGORIES_FILE,
                            StrategyType.FIRST, self.regr_yaml_file, prefix)
        rc = self.join_back_up()
        self.assertEqual(self.rc, rc)

    def test_something_lost_if_not_fit(self):
        prefix = os.path.join(self.target_prefix_file.name, 'split-')
        separate_categories(self.one_fewer_categories_file,
                            StrategyType.FIRST, self.regr_yaml_file, prefix, overflow_category=False)
        files = get_applicable_files_in(self.target_prefix_file.name)
        self.assertFalse(any('overflow-' in f for f in files))
        rc = self.join_back_up()
        self.assertLess(len(rc.cases), len(self.rc.cases))
        self.assertNotEqual(rc, self.rc)

    def test_something_written_in_overflow(self):
        prefix = os.path.join(self.target_prefix_file.name, 'split-')
        separate_categories(self.one_fewer_categories_file,
                            StrategyType.FIRST, self.regr_yaml_file, prefix, overflow_category=True)
        files = get_applicable_files_in(self.target_prefix_file.name)
        self.assertTrue(any('overflow-' in f for f in files))

    def test_something_NOT_lost_if_use_overflow(self):
        prefix = os.path.join(self.target_prefix_file.name, 'split-')
        separate_categories(self.one_fewer_categories_file,
                            StrategyType.FIRST, self.regr_yaml_file, prefix, overflow_category=True)
        rc = self.join_back_up()
        self.assertEqual(self.rc, rc)
