
from enum import Enum
from typing import Dict, Iterator, List, Optional, Set, Tuple, Type, Any, TypeVar, Union
import yaml
import logging
import tqdm

from pydantic import BaseModel

from medcat.cat import CAT
from medcat.cdb import CDB

from results import MultiDescriptor, ResultDescriptor

logger = logging.getLogger(__name__)

ENUM = TypeVar('ENUM', bound=Enum)


def loosely_match_enum(e_type: Type[ENUM], name: str) -> ENUM:
    """Loosely (i.e case-insensitively) match enum names.

    Args:
        e_type (Type[Enum]): The type of enum to use
        name (str): The case-insensitive name

    Raises:
        key_err: KeyError if the key is unable to be loosely matched

    Returns:
        Enum: The enum constant that was found
    """
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
    """Describes the filter strategy.
    I.e whether to match all or any
    of the filters specified.
    """
    ALL = 1
    """Specified that all filters must be satisfied"""
    ANY = 2
    """Specified that any of the filters must be satisfied"""

    @classmethod
    def match_str(cls, name: str) -> 'FilterStrategy':
        """Find a loose string match.

        Args:
            name (str): The name of the enum

        Returns:
            FilterStrategy: The matched FilterStrategy
        """
        return loosely_match_enum(FilterStrategy, name)


class TargetInfo:
    """The helper class to identify individual target info.
    The main reason for this class is to simplify type hints.

    Args:
        cui (str): The CUI of the target
        val (str): The name/value of the target
    """

    def __init__(self, cui: str, val: str) -> None:
        self.cui = cui
        self.val = val

    def __str__(self) -> str:
        return f'TI[{self.cui}:{self.val}]'

    def __repr__(self) -> str:
        return f'<{self}>'


class TranslationLayer:
    """The translation layer for translating:
    - CUIs to names
    - names to CUIs
    - type_ids to CUIs
    - CUIs to chil CUIs

    The idea is to decouple these translations from the CDB instance in case something changes there.

    Args:
        cui2names (Dict[str, Set[str]]): The map from CUI to names
        name2cuis (Dict[str, Set[str]]): The map from name to CUIs
        cui2type_ids (Dict[str, Set[str]]): The map from CUI to type_ids
        cui2children (Dict[str, Set[str]]): The map from CUI to child CUIs
    """

    def __init__(self, cui2names: Dict[str, Set[str]], name2cuis: Dict[str, Set[str]],
                 cui2type_ids: Dict[str, Set[str]], cui2children: Dict[str, Set[str]]) -> None:
        self.cui2names = cui2names
        self.name2cuis = name2cuis
        self.cui2type_ids = cui2type_ids
        self.cui2children = cui2children
        for cui in cui2names:
            if cui not in cui2children:
                self.cui2children[cui] = {}

    def targets_for(self, cui: str) -> Iterator[TargetInfo]:
        for name in self.cui2names[cui]:
            yield TargetInfo(cui, name)

    def all_targets(self) -> Iterator[TargetInfo]:
        """Get a generator of all target information objects.
        This is the starting point for checking cases.

        Yields:
            Iterator[TargetInfo]: The iterator of the target info
        """
        for cui, names in self.cui2names.items():
            for name in names:
                yield TargetInfo(cui, name)

    @classmethod
    def from_CDB(cls, cdb: CDB) -> 'TranslationLayer':
        """Construct a TranslationLayer object from a context database (CDB).

        This translation layer will refer to the same dicts that the CDB refers to.
        While there is no obvious reason these should be modified, it's something to keep in mind.

        Args:
            cdb (CDB): The CDB

        Returns:
            TranslationLayer: The subsequent TranslationLayer
        """
        return TranslationLayer(cdb.cui2names, cdb.name2cuis, cdb.cui2type_ids, cdb.addl_info['pt2ch'])


class FilterType(Enum):
    """The types of targets that can be specified
    """
    TYPE_ID = 1
    """Filters by specified type_ids"""
    CUI = 2
    """Filters by specified CUIs"""
    NAME = 3
    """Filters by specified names"""
    CUI_AND_CHILDREN = 4
    """Filter by CUI but also allow children, up to a specified distance"""

    @classmethod
    def match_str(cls, name: str) -> 'FilterType':
        """Case insensitive matching for FilterType

        Args:
            name (str): The naeme to be matched

        Returns:
            FilterType: The matched FilterType
        """
        return loosely_match_enum(FilterType, name)


class TypedFilter(BaseModel):
    """The base class for targeting.
    """
    type: FilterType

    def get_applicable_targets(self, translation: TranslationLayer, input: Iterator[TargetInfo]) -> Iterator[TargetInfo]:
        """Get the targets applicable for this filter.

        Args:
            translation (TranslationLayer): The translation layer
            input (Iterator[TargetInfo]): The input targets

        Yields:
            Iterator[TargetInfo]: The output targets
        """
        pass  # has to be overwritten

    @classmethod
    def one_from_input(cls, target_type: str, vals: Union[str, list, dict]) -> 'TypedFilter':
        """Get one typed filter from the input target type and values.
        The values can either a be a string for a single target,
        a list of strings for multiple targets, or
        a dict in some more complicated cases (i.e CUI_AND_CHILDREN).

        Args:
            target_type (str): The target type as string
            vals (Union[str, list, dict]): The values

        Raises:
            ValueError: If the values are malformed

        Returns:
            TypedFilter: The parsed filter
        """
        t_type: FilterType = FilterType.match_str(target_type)
        if isinstance(vals, list):
            filt = MultiFilter(type=t_type, values=vals)
        elif isinstance(vals, dict):
            if t_type != FilterType.CUI_AND_CHILDREN:
                # currently only applicable for CUI_AND_CHILDREN case
                raise ValueError(f'Misconfigured config for {target_type}, '
                                 'expected either a value or a list of values '
                                 'for this type of filter')
            depth = vals['depth']
            delegate = cls.one_from_input(target_type, vals['cui'])
            if t_type is FilterType.CUI_AND_CHILDREN:
                filt = CUIWithChildFilter(
                    type=t_type, delegate=delegate, depth=depth)
        else:
            filt = SingleFilter(type=t_type, value=vals)
        return filt

    @classmethod
    def from_dict(cls, input: Dict[str, Any]) -> List['TypedFilter']:
        """Construct a list of TypedFilter from a dict.

        The assumed structure is:
        {<filter type>: <filtered value>}
        or
        {<filter type>: [<filtered value2>, <filtered value 2>]}
        There can be multiple filter types defined.

        This creates instances MultiFilter and SingleFilter as needed.

        Returns:
            List[TypedFilter]: The list of constructed TypedFilter
        """
        parsed_targets: List[TypedFilter] = []
        for target_type, vals in input.items():
            filt = cls.one_from_input(target_type, vals)
            parsed_targets.append(filt)
        return parsed_targets


class FilterOptions(BaseModel):
    """A class describing the options for the filters
    """
    strategy: FilterStrategy
    onlyprefnames: bool = False

    @classmethod
    def from_dict(cls, section: Dict[str, str]) -> 'FilterOptions':
        """Construct a FilterOptions instance from a dict.

        The assumed structure is:
        {'strategy': <'all' or 'any'>,
        'prefname-only': 'true'}

        Both strategy and prefname-only are optional.

        Args:
            section (Dict[str, str]): The dict to parse

        Returns:
            FilterOptions: The resulting FilterOptions
        """
        if 'strategy' in section:
            strategy = FilterStrategy.match_str(section['strategy'])
        else:
            strategy = FilterStrategy.ALL  # default
        if 'prefname-only' in section:
            onlyprefnames = section['prefname-only'].lower() == 'true'
        else:
            onlyprefnames = False
        return FilterOptions(strategy=strategy, onlyprefnames=onlyprefnames)


class SingleFilter(TypedFilter):
    """A filter with a single value to filter against.
    """
    value: str

    def get_applicable_targets(self, translation: TranslationLayer, in_gen: Iterator[TargetInfo]) -> Iterator[TargetInfo]:
        """Get all applicable targets for this filter

        Args:
            translation (TranslationLayer): The translation layer
            in_gen (Iterator[TargetInfo]): The input generator / iterator

        Yields:
            Iterator[TargetInfo]: The output generator
        """
        if self.type == FilterType.CUI or self.type == FilterType.CUI_AND_CHILDREN:
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
    """A filter with multiple values to filter against.
    """
    values: List[str]

    def get_applicable_targets(self, translation: TranslationLayer, in_gen: Iterator[TargetInfo]) -> Iterator[TargetInfo]:
        """Get all applicable targets for this filter

        Args:
            translation (TranslationLayer): The translation layer
            in_gen (Iterator[TargetInfo]): The input generator / iterator

        Yields:
            Iterator[TargetInfo]: The output generator
        """
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


class CUIWithChildFilter(TypedFilter):
    delegate: TypedFilter
    depth: int

    def get_applicable_targets(self, translation: TranslationLayer, in_gen: Iterator[TargetInfo]) -> Iterator[TargetInfo]:
        """Get all applicable targets for this filter

        Args:
            translation (TranslationLayer): The translation layer
            in_gen (Iterator[TargetInfo]): The input generator / iterator

        Yields:
            Iterator[TargetInfo]: The output generator
        """
        for ti in self.delegate.get_applicable_targets(translation, in_gen):
            yield ti
            yield from self.get_children_of(translation, ti.cui, cur_depth=1)

    def get_children_of(self, translation: TranslationLayer, cui: str, cur_depth: int) -> Iterator[TargetInfo]:
        for child in translation.cui2children[cui]:
            yield from translation.targets_for(child)
            if cur_depth < self.depth:
                yield from self.get_children_of(translation, child, cur_depth=cur_depth + 1)


class RegressionCase(BaseModel):
    """A regression case that has a name, defines options, filters and phrases.s
    """
    name: str
    options: FilterOptions
    filters: List[TypedFilter]
    phrases: List[str]
    report: Optional[ResultDescriptor] = None

    def get_all_targets(self, in_set: Iterator[TargetInfo], translation: TranslationLayer) -> Iterator[TargetInfo]:
        """Get all applicable targets for this regression case

        Args:
            in_set (Iterator[TargetInfo]): The input generator / iterator
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[TargetInfo]: The output generator
        """
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

    def check_specific_for_phrase(self, cat: CAT, ti: TargetInfo, phrase: str) -> bool:
        """Checks whether the specific target along with the specified phrase
        is able to be identified using the specified model.

        Args:
            cat (CAT): The model
            ti (TargetInfo): The target info
            phrase (str): The phrase to check

        Returns:
            bool: Whether or not the target was correctly identified
        """
        res = cat.get_entities(phrase % ti.val, only_cui=True)
        ents = res['entities']
        found_cuis = [ents[nr] for nr in ents]
        success = ti.cui in found_cuis
        if success:
            logger.debug(
                'Matched test case %s in phrase "%s"', ti, phrase)
        else:
            logger.debug(
                'FAILED to match test case %s in phrase "%s", found the following CUIS: %s', ti, phrase, found_cuis)
        if self.report is not None:
            self.report.report(ti.cui, ti.val, phrase, success)
        return success

    def get_all_subcases(self, translation: TranslationLayer) -> Iterator[Tuple[TargetInfo, str]]:
        """Get all subcases for this case.
        That is, all combinations of targets with their appropriate phrases.

        Args:
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[Tuple[TargetInfo, str]]: The generator for the target info and the phrase
        """
        for ti in self.get_all_targets(translation.all_targets(), translation):
            for phrase in self.phrases:
                yield ti, phrase

    def check_case(self, cat: CAT, translation: TranslationLayer) -> Tuple[int, int]:
        """Check the regression case against a model.
        I.e check all its applicable targets.

        Args:
            cat (CAT): The CAT instance
            translation (TranslationLayer): The translation layer

        Returns:
            Tuple[int, int]: Number of successes and number of failures
        """
        success = 0
        fail = 0
        for target, phrase in self.get_all_subcases(translation):
            if self.check_specific_for_phrase(cat, target, phrase):
                success += 1
            else:
                fail += 1
        return success, fail

    @classmethod
    def from_dict(cls, name: str, in_dict: dict) -> 'RegressionCase':
        """Construct the regression case from a dict.

        The expected stucture:
        {
            'targeting': {
                'strategy': 'ALL', # optional
                'prefname-only': 'false', # optional
                'filters': {
                    <filter type>: <filter values>, # possibly multiple
                }
            },
            'phrases': ['phrase %s'] # possible multiple
        }

        Parsing the different parts of are delegated to
        other methods within the relevant classes.
        Delegators include: FilterOptions, TypedFilter

        Args:
            name (str): The name of the case
            in_dict (dict): The dict describing the case

        Raises:
            ValueError: If the input dict does not have the 'targeting' section
            ValueError: If the 'targeting' section does not have a 'filters' section
            ValueError: If there are no phrases defined

        Returns:
            RegressionCase: The constructed regression case
        """
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
    """The regression checker.
    This is used to check a bunch of regression cases at once against a model.

    Args:
        cases (List[RegressionCase]): The list of regression cases
        use_report (bool): Whether or not to use the report functionality (defaults to False)
    """

    def __init__(self, cases: List[RegressionCase], use_report: bool = True) -> None:
        self.cases: List[RegressionCase] = cases
        self.use_report = use_report
        self.report: Optional[MultiDescriptor] = None if not self.use_report else MultiDescriptor(
            name='ALL')  # TODO - allow setting names
        if self.report is not None:
            for case in self.cases:
                cur_rd = ResultDescriptor(name=case.name)
                self.report.parts.append(cur_rd)
                case.report = cur_rd

    def get_all_subcases(self, translation: TranslationLayer) -> Iterator[Tuple[RegressionCase, TargetInfo, str]]:
        """Get all subcases (i.e regssion case, target info and phrase) for this checker.

        Args:
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[Tuple[RegressionCase, TargetInfo, str]]: The generator for all the cases
        """
        for case in self.cases:
            for ti, phrase in case.get_all_subcases(translation):
                yield case, ti, phrase

    def check_model(self, cat: CAT, translation: TranslationLayer,
                    total: Optional[int] = None) -> Union[Tuple[int, int], MultiDescriptor]:
        """_summary_

        Args:
            cat (CAT): The model to check against
            translation (TranslationLayer): The translation layer
            total (Optional[int]): The total number of (sub)cases expected (for a progress bar)

        Returns:
            Union[Tuple[int, int], MultiDescriptor]: The number of successful and failed checks,
                                                        or a MultiDescriptor if a report was requested
        """
        successes, fails = 0, 0
        if total is not None:
            for case, ti, phrase in tqdm.tqdm(self.get_all_subcases(translation), total=total):
                if case.check_specific_for_phrase(cat, ti, phrase):
                    successes += 1
                else:
                    fails += 1
        else:
            for case, ti, phrase in self.get_all_subcases(translation):
                if case.check_specific_for_phrase(cat, ti, phrase):
                    successes += 1
                else:
                    fails += 1
        if self.use_report:
            return self.report
        return successes, fails

    def __str__(self) -> str:
        return f'RegressionTester[cases={self.cases}]'

    def __repr__(self) -> str:
        return f'<{self}>'

    @classmethod
    def from_dict(cls, in_dict: dict) -> 'RegressionChecker':
        """Construct a RegressionChecker from a dict.

        Most of the parsing is handled in RegressionChecker.from_dict.
        This just assumes that each key in the dict is a name
        and each value describes a RegressionCase.

        Args:
            in_dict (dict): The input dict

        Returns:
            RegressionChecker: The built regression checker
        """
        cases = []
        for case_name, details in in_dict.items():
            case = RegressionCase.from_dict(case_name, details)
            cases.append(case)
        return RegressionChecker(cases=cases)

    @classmethod
    def from_yaml(cls, file_name: str) -> 'RegressionChecker':
        """Constructs a RegressionChcker from a YAML file.

        The from_dict method is used for the construction from the dict.

        Args:
            file_name (str): The file name

        Returns:
            RegressionChecker: The constructed regression checker
        """
        with open(file_name, 'r') as f:
            data = yaml.safe_load(f)
        return RegressionChecker.from_dict(data)
