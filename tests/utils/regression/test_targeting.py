from typing import Optional, List
from unittest import TestCase

from medcat.config import Config
from medcat.utils.regression import targeting

from collections import defaultdict
from copy import deepcopy


class FakeCDB:

    def __init__(self, def_name: str, def_cui: str, pt2ch: Optional[dict] = None) -> None:
        self.cui2names = defaultdict(lambda: {def_name})
        self.name2cuis = defaultdict(lambda: {def_cui})
        self.cui2type_ids = {}  # NOTE: shouldn't be needed
        if pt2ch is None:
            pt2ch = {}
        self.addl_info = {'pt2ch': pt2ch}
        self.config = Config()

    def copy(self) -> 'FakeCDB':
        cui2names = deepcopy(self.cui2names)
        name2cuis = deepcopy(self.name2cuis)
        addl_info = deepcopy(self.addl_info)
        copy = FakeCDB(cui2names[None], name2cuis[None])
        copy.cui2names = cui2names
        copy.name2cuis = name2cuis
        copy.addl_info = addl_info
        return copy

    @property
    def cui2preferred_name(self) -> dict:
        return {cui: list(names)[0] for cui, names in self.cui2names.items()}


class OptionSetTests(TestCase):
    OPTIONSET_SIMPLE = {
        'placeholders': [
            {
                'placeholder': '%s',
                'cuis': ['CUI1']
            }
        ]
    }
    OPTIONSET_MULTI = {
        'placeholders': [
            {
                'placeholder': '%s',
                'cuis': ['CUI1']
            },
            {
                'placeholder': '[PH1]',
                'cuis': ['CUI2']
            },
        ]
    }
    ALL_WORKING = [OPTIONSET_SIMPLE, OPTIONSET_MULTI]
    OPTIONSET_MULTI_SAMES = {
        'placeholders': OPTIONSET_SIMPLE['placeholders'] * 2
    }
    OPTIONSET_0_PH = {'placeholders': []}
    OPTIONSET_NO_PH = {'SomeJunk': [{'KEYS': 'VALUES'}]}
    EXPECTED_TARGETS = [
        (OPTIONSET_SIMPLE, 1),
        (OPTIONSET_MULTI, 2)
    ]
    ALL_ALL = ALL_WORKING + [OPTIONSET_MULTI_SAMES, OPTIONSET_0_PH, OPTIONSET_NO_PH]
    cdb = FakeCDB('NAME', 'CUI1')

    @classmethod
    def discover_cuis_for(cls, d: dict) -> list:
        all_cuis = []
        phs = d.get('placeholders', [])
        for ph in phs:
            all_cuis.extend(ph.get('cuis', []))
        return all_cuis


    @classmethod
    def discover_all_used_cuis(cls) -> list:
        all_cuis = []
        for d in cls.ALL_ALL:
            all_cuis.extend(cls.discover_cuis_for(d))
        return all_cuis

    @classmethod
    def setUpClass(cls) -> None:
        # add name per CUI
        for cui in cls.discover_all_used_cuis():
            cls.cdb.cui2names[cui] = {f'cui-{cui}-name'}
        cls.tl = targeting.TranslationLayer.from_CDB(cls.cdb)

    def test_create_from_dict_simple(self):
        os = targeting.OptionSet.from_dict(self.OPTIONSET_SIMPLE)
        self.assertIsInstance(os, targeting.OptionSet)

    def test_create_from_dict_multi(self):
        os = targeting.OptionSet.from_dict(self.OPTIONSET_MULTI)
        self.assertIsInstance(os, targeting.OptionSet)

    def test_creation_fails_with_same_placeholders(self):
        with self.assertRaises(targeting.ProblematicOptionSetException):
            targeting.OptionSet.from_dict(self.OPTIONSET_MULTI_SAMES)

    def test_creation_fails_no_placeholders(self):
        with self.assertRaises(targeting.ProblematicOptionSetException):
            targeting.OptionSet.from_dict(self.OPTIONSET_NO_PH)

    def test_creation_fails_0_placeholders(self):
        with self.assertRaises(targeting.ProblematicOptionSetException):
            targeting.OptionSet.from_dict(self.OPTIONSET_0_PH)

    def test_get_placeholders(self):
        for nr, target in enumerate(self.ALL_WORKING):
            with self.subTest(f'Target nr {nr}'):
                os = targeting.OptionSet.from_dict(target)
                self.assertEqual(len(os.options), len(target['placeholders']))

    def test_uses_default_allow_any(self):
        _def_value = targeting.OptionSet(options=[]).allow_any_combinations
        for nr, target in enumerate(self.ALL_WORKING):
            with self.subTest(f'Target nr {nr}'):
                os = targeting.OptionSet.from_dict(target)
                self.assertEqual(os.allow_any_combinations, _def_value)

    def test_gets_correct(self):
        for nr, (d, num_of_targets) in enumerate(self.EXPECTED_TARGETS):
            with self.subTest(f"Part: {nr}"):
                os = targeting.OptionSet.from_dict(d)
                targets = list(os.get_preprocessors_and_targets(self.tl))
                self.assertEqual(len(targets), num_of_targets)


class OnePerNameOptionSetTests(TestCase):
    SIMPLE = OptionSetTests.OPTIONSET_SIMPLE
    MULTI_PLACEHOLDER_MULTI_CUI_ONLY_ORDERED = {
        'placeholders': [
            {
                'placeholder': '[PH1]',
                'cuis': ['CUI_11', 'CUI_12']
            },
            {
                'placeholder': '[PH2]',
                'cuis': ['CUI_21', 'CUI_22']
            },
            {
                'placeholder': '[PH3]',
                'cuis': ['CUI_31', 'CUI_32']
            }
        ],
        'any-combination': False
    }
    MULTI_PLACEHOLDER_MULTI_CUI_ANY_COMB = {**MULTI_PLACEHOLDER_MULTI_CUI_ONLY_ORDERED,
                                            'any-combination': True}
    MULTI_PLACEHOLDER_MULTI_CUI_ONLY_ORDERED_BROKEN = deepcopy(MULTI_PLACEHOLDER_MULTI_CUI_ONLY_ORDERED)

    @classmethod
    def setUpClass(cls) -> None:
        # remove a CUI so it's breokn
        cls.MULTI_PLACEHOLDER_MULTI_CUI_ONLY_ORDERED_BROKEN['placeholders'][0]['cuis'] = ['CUI11']
        cuis = OptionSetTests.discover_cuis_for(cls.SIMPLE)
        cdb = FakeCDB('NAME', 'CUI1')
        total_names_simple = 0
        for cui in cuis:
            cdb.cui2names[cui].add(f"CUi-name-2-={cui}")
            total_names_simple += len(cdb.cui2names[cui])
        cls.cdb = cdb
        cls.tl = targeting.TranslationLayer.from_CDB(cls.cdb)
        cls.total_names_simple = total_names_simple
        for cui in OptionSetTests.discover_cuis_for(cls.MULTI_PLACEHOLDER_MULTI_CUI_ONLY_ORDERED):
            cdb.cui2names[cui] = {f'CUI=name-4-{cui}'}

    def test_uneven_multi_fails(self):
        with self.assertRaises(targeting.ProblematicOptionSetException):
            targeting.OptionSet.from_dict(self.MULTI_PLACEHOLDER_MULTI_CUI_ONLY_ORDERED_BROKEN)

    def test_even_builds(self):
        os = targeting.OptionSet.from_dict(self.MULTI_PLACEHOLDER_MULTI_CUI_ONLY_ORDERED)
        self.assertIsInstance(os, targeting.OptionSet)
        self.assertFalse(os.allow_any_combinations)

    def test_any_order_builds(self):
        os = targeting.OptionSet.from_dict(self.MULTI_PLACEHOLDER_MULTI_CUI_ANY_COMB)
        self.assertIsInstance(os, targeting.OptionSet)
        self.assertTrue(os.allow_any_combinations)

    def test_even_has_a_few_targets(self):
        os = targeting.OptionSet.from_dict(self.MULTI_PLACEHOLDER_MULTI_CUI_ONLY_ORDERED)
        targets = list(os.get_preprocessors_and_targets(self.tl))
        # 2 for each of the 3 PRIMARY options
        self.assertEqual(len(targets), 2 * 3)

    def assert_all_unique(self, targets: List[tuple]):
        for nr1, ctarget in enumerate(targets):
            for nr2, other in enumerate(targets[nr1 + 1:]):
                with self.subTest(f"{nr1}x{nr2}"):
                    self.assertNotEqual(ctarget, other)
                    self.assertTrue(any(cpart != opart for cpart, opart in zip(ctarget, other)))

    def test_even_has_unique_targets(self):
        os = targeting.OptionSet.from_dict(self.MULTI_PLACEHOLDER_MULTI_CUI_ONLY_ORDERED)
        targets = list(os.get_preprocessors_and_targets(self.tl))
        self.assert_all_unique(targets)

    def test_any_order_has_many_targets(self):
        os = targeting.OptionSet.from_dict(self.MULTI_PLACEHOLDER_MULTI_CUI_ANY_COMB)
        targets = list(os.get_preprocessors_and_targets(self.tl))
        # for each of the 3 PRIMARY options, the combinations of all
        self.assertEqual(len(targets), 3 * 2 ** 3)

    def test_any_order_has_unique_targets(self):
        os = targeting.OptionSet.from_dict(self.MULTI_PLACEHOLDER_MULTI_CUI_ANY_COMB)
        targets = list(os.get_preprocessors_and_targets(self.tl))
        self.assert_all_unique(targets)
