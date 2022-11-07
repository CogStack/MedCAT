

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union, cast
import yaml
import logging
import tqdm

from pydantic import BaseModel

from medcat.cat import CAT
from medcat.utils.regression.targeting import CUIWithChildFilter, FilterOptions, FilterType, TypedFilter, TranslationLayer, FilterStrategy

from medcat.utils.regression.results import FailDescriptor, MultiDescriptor, ResultDescriptor

logger = logging.getLogger(__name__)


class RegressionCase(BaseModel):
    """A regression case that has a name, defines options, filters and phrases.s
    """
    name: str
    options: FilterOptions
    filters: List[TypedFilter]
    phrases: List[str]
    report: Optional[ResultDescriptor] = None

    def get_all_targets(self, in_set: Iterator[Tuple[str, str]], translation: TranslationLayer) -> Iterator[Tuple[str, str]]:
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

    def check_specific_for_phrase(self, cat: CAT, cui: str, name: str, phrase: str,
                                  translation: TranslationLayer) -> bool:
        """Checks whether the specific target along with the specified phrase
        is able to be identified using the specified model.

        Args:
            cat (CAT): The model
            ti (TargetInfo): The target info
            phrase (str): The phrase to check
            translation (TranslationLayer): The translation layer

        Returns:
            bool: Whether or not the target was correctly identified
        """
        res = cat.get_entities(phrase % name, only_cui=False)
        ents = res['entities']
        found_cuis = [ents[nr]['cui'] for nr in ents]
        success = cui in found_cuis
        fail_reason: Optional[FailDescriptor]
        if success:
            logger.debug(
                'Matched test case %s in phrase "%s"', (cui, name), phrase)
            fail_reason = None
        else:
            fail_reason = FailDescriptor.get_reason_for(cui, name, res,
                                                        translation)
            found_names = [ents[nr]['source_value'] for nr in ents]
            cuis_names = ', '.join([f'{fcui}|{fname}'
                                    for fcui, fname in zip(found_cuis, found_names)])
            logger.debug(
                'FAILED to match (%s) test case %s in phrase "%s", '
                'found the following CUIS/names: %s', fail_reason, (cui, name), phrase, cuis_names)
        if self.report is not None:
            self.report.report(cui, name, phrase,
                               success, fail_reason)
        return success

    def _get_all_cuis_names_types(self) -> Tuple[Set[str], Set[str], Set[str]]:
        cuis = set()
        names = set()
        types = set()
        for filt in self.filters:
            if filt.type == FilterType.CUI:
                cuis.update(filt.values)
            elif filt.type == FilterType.CUI_AND_CHILDREN:
                cuis.update(cast(CUIWithChildFilter, filt).delegate.values)
            if filt.type == FilterType.NAME:
                names.update(filt.values)
            if filt.type == FilterType.TYPE_ID:
                types.update(filt.values)
        return cuis, names, types

    def get_all_subcases(self, translation: TranslationLayer) -> Iterator[Tuple[str, str, str]]:
        """Get all subcases for this case.
        That is, all combinations of targets with their appropriate phrases.

        Args:
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[Tuple[TargetInfo, str]]: The generator for the target info and the phrase
        """
        cntr = 0
        for cui, name in self.get_all_targets(translation.all_targets(*self._get_all_cuis_names_types()), translation):
            for phrase in self.phrases:
                cntr += 1
                yield cui, name, phrase
        if not cntr:
            for cui, name in self._get_specific_cui_and_name():
                for phrase in self.phrases:
                    yield cui, name, phrase

    def _get_specific_cui_and_name(self) -> Iterator[Tuple[str, str]]:
        if len(self.filters) != 2:
            return
        if self.options.strategy != FilterStrategy.ALL:
            return
        f1, f2 = self.filters
        if f1.type == FilterType.NAME and f2.type == FilterType.CUI:
            name_filter, cui_filter = f1, f2
        elif f2.type == FilterType.NAME and f1.type == FilterType.CUI:
            name_filter, cui_filter = f2, f1
        else:
            return
        # There should only ever be one for the ALL strategty
        # because otherwise a match is impossible
        for name in name_filter.values:
            for cui in cui_filter.values:
                yield cui, name

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
        for cui, name, phrase in self.get_all_subcases(translation):
            if self.check_specific_for_phrase(cat, cui, name, phrase, translation):
                success += 1
            else:
                fail += 1
        return success, fail

    def to_dict(self) -> dict:
        """Converts the RegressionCase to a dict for serialisation.

        Returns:
            dict: The dict representation
        """
        d: Dict[str, Any] = {'phrases': list(self.phrases)}
        targeting = self.options.to_dict()
        targeting['filters'] = {}
        for filt in self.filters:
            targeting['filters'].update(filt.to_dict())
        d['targeting'] = targeting
        return d

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

    def get_all_subcases(self, translation: TranslationLayer) -> Iterator[Tuple[RegressionCase, str, str, str]]:
        """Get all subcases (i.e regssion case, target info and phrase) for this checker.

        Args:
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[Tuple[RegressionCase, TargetInfo, str]]: The generator for all the cases
        """
        for case in self.cases:
            for cui, name, phrase in case.get_all_subcases(translation):
                yield case, cui, name, phrase

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
                if case.check_specific_for_phrase(cat, ti, phrase, translation):
                    successes += 1
                else:
                    fails += 1
        else:
            for case in tqdm.tqdm(self.cases):
                for ti, phrase in case.get_all_subcases(translation):
                    if case.check_specific_for_phrase(cat, ti, phrase, translation):
                        successes += 1
                    else:
                        fails += 1
        if self.use_report and self.report is not None:
            return self.report
        return successes, fails

    def __str__(self) -> str:
        return f'RegressionTester[cases={self.cases}]'

    def __repr__(self) -> str:
        return f'<{self}>'

    def to_dict(self) -> dict:
        """Converts the RegressionChecker to dict for serialisation.

        Returns:
            dict: The dict representation
        """
        d = {}
        for case in self.cases:
            d[case.name] = case.to_dict()
        return d

    def to_yaml(self) -> str:
        """Convert the RegressionChecker to YAML string.

        Returns:
            str: The YAML representation
        """
        return yaml.dump(self.to_dict())

    def __eq__(self, other: object) -> bool:
        # only checks cases
        if not isinstance(other, RegressionChecker):
            return False
        return self.cases == other.cases

    @classmethod
    def from_dict(cls, in_dict: dict, report: bool = False) -> 'RegressionChecker':
        """Construct a RegressionChecker from a dict.

        Most of the parsing is handled in RegressionChecker.from_dict.
        This just assumes that each key in the dict is a name
        and each value describes a RegressionCase.

        Args:
            in_dict (dict): The input dict
            report (bool): Whether or not to use a more comprehensive report (defaults to False)

        Returns:
            RegressionChecker: The built regression checker
        """
        cases = []
        for case_name, details in in_dict.items():
            case = RegressionCase.from_dict(case_name, details)
            cases.append(case)
        return RegressionChecker(cases=cases, use_report=report)

    @classmethod
    def from_yaml(cls, file_name: str, report: bool = False) -> 'RegressionChecker':
        """Constructs a RegressionChcker from a YAML file.

        The from_dict method is used for the construction from the dict.

        Args:
            file_name (str): The file name
            report (bool): Whether or not to use a more comprehensive report (defaults to False)

        Returns:
            RegressionChecker: The constructed regression checker
        """
        with open(file_name, 'r') as f:
            data = yaml.safe_load(f)
        return RegressionChecker.from_dict(data, report=report)
