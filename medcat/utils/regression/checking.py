
from enum import Enum
from typing import Dict, Iterator, List, Set, Tuple, Type, Any
import yaml
import logging

from pydantic import BaseModel

from medcat.cat import CAT
from medcat.cdb import CDB


logger = logging.getLogger(__name__)


def loosely_match_enum(e_type: Type[Enum], name: str) -> Enum:
    _key_err = None
    try:
        return e_type[name]
    except KeyError as key_err:
        _key_err = key_err
    name = name.lower()
    try:
        return e_type[name]
    except KeyError:
        pass
    name = name.upper()
    try:
        return e_type[name]
    except KeyError:
        pass
    raise _key_err


class FilterStrategy(Enum):
    ALL = 1
    all = 1
    ANY = 2
    any = 2

    @classmethod
    def match_str(cls, name: str) -> 'FilterStrategy':
        return loosely_match_enum(FilterStrategy, name)


class TargetInfo:

    def __init__(self, cui: str, val: str) -> None:
        self.cui = cui
        self.val = val

    def __str__(self) -> str:
        return f'TI[{self.cui}:{self.val}]'

    def __repr__(self) -> str:
        return f'<{self}>'


class TranslationLayer:

    def __init__(self, cui2names: Dict[str, Set[str]], name2cuis: Dict[str, Set[str]], cui2type_ids: Dict[str, Set[str]]) -> None:
        self.cui2names = cui2names
        self.name2cuis = name2cuis
        self.cui2type_ids = cui2type_ids

    def all_targets(self) -> Iterator[TargetInfo]:
        for cui, names in self.cui2names.items():
            for name in names:
                yield TargetInfo(cui, name)

    @classmethod
    def from_CDB(cls, cdb: CDB) -> 'TranslationLayer':
        # TODO - these are now referring to the same objects
        # so changing these within here will change the originals
        # we might want to create copies instead?
        return TranslationLayer(cdb.cui2names, cdb.name2cuis, cdb.cui2type_ids)


class FilterType(Enum):
    TYPE_ID = 1
    type_id = 1
    CUI = 2
    cui = 2
    NAME = 3
    name = 3

    @classmethod
    def match_str(cls, name: str) -> 'FilterType':
        return loosely_match_enum(FilterType, name)


class TypedFilter(BaseModel):
    type: FilterType

    def get_applicable_targets(self, translation: TranslationLayer, input: Iterator[TargetInfo]) -> Iterator[TargetInfo]:
        pass  # has to be overwritten

    @classmethod
    def from_dict(cls, input: Dict[str, Any]) -> List['TypedFilter']:
        parsed_targets = []
        for target_type, vals in input.items():
            t_type: FilterType = FilterType.match_str(target_type)
            if isinstance(vals, list):
                parsed_targets.append(MultiFilter(type=t_type, values=vals))
            else:
                parsed_targets.append(SingleFilter(type=t_type, value=vals))
        return parsed_targets


class FilterOptions(BaseModel):
    strategy: FilterStrategy
    onlyprefnames: bool = False

    @classmethod
    def from_dict(cls, section: Dict[str, str]) -> 'FilterOptions':
        if 'strategy' in section:
            strategy = FilterStrategy.match_str(section['strategy'])
        else:
            strategy = FilterStrategy.ALL  # default
        if 'prefname-only' in section:
            onlyprefnames = bool(section['prefname-only'])
        else:
            onlyprefnames = False
        return FilterOptions(strategy=strategy, onlyprefnames=onlyprefnames)


class SingleFilter(TypedFilter):
    value: str

    def get_applicable_targets(self, translation: TranslationLayer, in_gen: Iterator[TargetInfo]) -> Iterator[TargetInfo]:
        if self.type == FilterType.CUI:
            for ti in in_gen:
                if ti.cui == self.value:
                    yield ti
        if self.type == FilterType.NAME:
            for ti in in_gen:
                if self.value in ti.val:
                    yield ti
        if self.type == FilterType.TYPE_ID:
            for ti in in_gen:
                if ti.cui in translation.cui2type_ids and self.value in translation.cui2type_ids[ti.cui]:
                    yield ti


class MultiFilter(TypedFilter):
    values: List[str]

    def get_applicable_targets(self, translation: TranslationLayer, in_gen: Iterator[TargetInfo]) -> Iterator[TargetInfo]:
        if self.type == FilterType.CUI:
            for ti in in_gen:
                if ti.cui in self.values:
                    yield ti
        if self.type == FilterType.NAME:
            for ti in in_gen:
                if ti.val in self.values:
                    yield ti
        if self.type == FilterType.TYPE_ID:
            for ti in in_gen:
                if ti.cui in translation.cui2type_ids and translation.cui2type_ids[ti.cui] in self.values:
                    yield ti


class RegressionCase(BaseModel):
    name: str
    options: FilterOptions
    filters: List[TypedFilter]
    phrases: List[str]

    def get_all_targets(self, in_set: Iterator[TargetInfo], translation: TranslationLayer) -> Iterator[TargetInfo]:
        if len(self.filters) == 1:
            yield from self.filters[0].get_applicable_targets(translation, in_set)
            return
        if self.options.strategy == FilterStrategy.ANY:
            for filter in self.filters:
                yield from filter.get_applicable_targets(translation, in_set)
        elif self.options.strategy == FilterStrategy.ALL:
            cur_gen = in_set
            for filter in self.filters:
                cur_gen = filter.get_applicable_targets(translation, cur_gen)
            yield from cur_gen

    def check_case(self, cat: CAT, translation: TranslationLayer) -> Tuple[int, int]:
        success = 0
        fail = 0
        for ti in self.get_all_targets(translation.all_targets(), translation):
            for phrase in self.phrases:
                res = cat.get_entities(phrase % ti.val, only_cui=True)
                ents = res['entities']
                found_cuis = [ents[nr]['cui'] for nr in ents]
                if ti.cui in found_cuis:
                    logger.debug(
                        'Matched test case %s in phrase "%s"', ti, phrase)
                    success += 1
                else:
                    logger.debug(
                        'FAILED to match test case %s in phrase "%s", found the following CUIS: %s', ti, phrase, found_cuis)
                    fail += 1
        return success, fail

    @classmethod
    def from_dict(cls, name: str, in_dict: dict) -> 'FilterStrategy':
        # set up targeting
        if 'targeting' not in in_dict:
            raise ValueError('Input dict should define targeting')
        targeting_section = in_dict['targeting']
        # set up options
        options = FilterOptions.from_dict(targeting_section)
        if 'filters' not in targeting_section:
            raise ValueError(
                'Input dict should have define targets section under targeting')
        # set up targets
        parsed_filters: List[TypedFilter] = TypedFilter.from_dict(
            targeting_section['filters'])
        # set up test phrases
        if 'phrases' not in in_dict:
            raise ValueError('Input dict should defined phrases')
        phrases = in_dict['phrases']
        if not isinstance(phrases, list):
            phrases = [phrases]  # just one defined
        if not phrases:
            raise ValueError('Need at least one target phrase')
        return RegressionCase(name=name, options=options, filters=parsed_filters, phrases=phrases)


class RegressionChecker:

    def __init__(self, cases: List[RegressionCase]) -> None:
        self.cases: List[RegressionCase] = cases

    def check_model(self, cat: CAT, translation: TranslationLayer):
        successes, fails = 0, 0
        for case in self.cases:
            s, f = case.check_case(cat, translation)
            successes += s
            fails += f
        return s, f

    def __str__(self) -> str:
        return f'RegressionTester[cases={self.cases}]'

    def __repr__(self) -> str:
        return f'<{self}>'

    @classmethod
    def from_dict(cls, in_dict: dict) -> 'RegressionChecker':
        cases = []
        for case_name, details in in_dict.items():
            case = RegressionCase.from_dict(case_name, details)
            cases.append(case)
        return RegressionChecker(cases=cases)


def read_options(file_name: str) -> None:
    with open(file_name, 'r') as f:
        data = yaml.safe_load(f)
    return RegressionChecker.from_dict(data)
