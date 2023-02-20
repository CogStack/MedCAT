from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, cast
import yaml
import logging
import tqdm
import datetime

from pydantic import BaseModel, Field

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
    report: ResultDescriptor

    def get_all_targets(self, in_set: Iterator[Tuple[str, str]], translation: TranslationLayer) -> Iterator[Tuple[str, str]]:
        """Get all applicable targets for this regression case

        Args:
            in_set (Iterator[Tuple[str, str]]): The input generator / iterator
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[Tuple[str, str]]: The output generator
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
            cui (str): The target CUI
            name (str): The target name
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
            Iterator[Tuple[str, str, str]]: The generator for the target info and the phrase
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
        return RegressionCase(name=name, options=options, filters=parsed_filters,
                              phrases=phrases, report=ResultDescriptor(name=name))

    def __hash__(self) -> int:
        return hash(str(self.to_dict()))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RegressionCase):
            return False
        return self.to_dict() == other.to_dict()


UNKNOWN_METADATA = 'Unknown'


def get_ontology_and_version(model_card: dict) -> Tuple[str, str]:
    """Attempt to get ontology (and its version) from a model card dict.

    If no ontology is found, 'Unknown' is returned.
    The version is always returned as the first source ontology.
    That is, unless the specified location does not exist in the model card,
    in which case 'Unknown' is returned.

    The ontology is assumed to be descibed at:
        model_card['Source Ontology'][0] (or model_card['Source Ontology'] if it's a string instead of a list)

    The ontology version is read from:
        model_card['Source Ontology'][0] (or model_card['Source Ontology'] if it's a string instead of a list)

    Currently, only SNOMED-CT, UMLS and ICD are supported / found.

    Args:
        model_card (dict): The input model card.

    Returns:
        str: The ontology (if found) or 'Unknown'
    Returns:
        Tuple[str, str]: The ontology (if found) or 'Unknown'; and the version (if found) or 'Unknown'
    """
    try:
        ont_list = model_card['Source Ontology']
        if isinstance(ont_list, list):
            ont1 = ont_list[0]
        elif isinstance(ont_list, str):
            ont1 = ont_list
        else:
            raise KeyError(f"Unknown source ontology: {ont_list}")
    except KeyError as key_err:
        logger.warn(
            "Didn't find the expected source ontology from the model card!", exc_info=key_err)
        return UNKNOWN_METADATA, UNKNOWN_METADATA
    # find ontology
    if 'SNOMED' in ont1.upper():
        return 'SNOMED-CT', ont1
    elif 'UMLS' in ont1.upper():
        return 'UMLS', ont1
    elif 'ICD' in ont1.upper():
        return 'ICD', ont1
    else:
        return UNKNOWN_METADATA, ont1


class MetaData(BaseModel):
    """The metadat for the regression suite.

    This should define which ontology (e.g UMLS or SNOMED) as well as
    which version was used when generating the regression suite.

    The metadata may contain further information as well, this may include
    the annotator(s) involved when converting from MCT export or other relevant data.
    """
    ontology: str
    ontology_version: str
    extra: dict = {}
    regr_suite_creation_date: str = Field(
        default_factory=lambda: str(datetime.datetime.now()))

    @classmethod
    def from_modelcard(cls, model_card: dict) -> 'MetaData':
        """Generate a MetaData object from a model card.

        This involves reading ontology info and version from the model card.

        It must be noted that the model card should be provided as a dict not a string.

        Args:
            model_card (dict): The CAT modelcard

        Returns:
            MetaData: The resulting MetaData
        """
        ontology, ont_version = get_ontology_and_version(model_card)
        return MetaData(ontology=ontology, ontology_version=ont_version, extra=model_card)

    @classmethod
    def unknown(self) -> 'MetaData':
        return MetaData(ontology=UNKNOWN_METADATA, ontology_version=UNKNOWN_METADATA,

                        extra={}, regr_suite_creation_date=UNKNOWN_METADATA)


def fix_np_float64(d: dict) -> None:
    """Fix numpy.float64 in dictrionary for yaml saving purposes.

    These types of objects are unable to be cleanly serialized using yaml.
    So we need to conver them to the corresponding floats.

    The changes will be made within the dictionary itself
    as well as dictionaries within, recursively.

    Args:
        d (dict): The input dict
        prefix (str, optional): The prefix for t. Defaults to ''.
    """
    import numpy as np
    for k, v in d.items():
        if isinstance(v, np.float64):
            d[k] = float(v)
        if isinstance(v, dict):
            fix_np_float64(v)


class RegressionChecker:
    """The regression checker.
    This is used to check a bunch of regression cases at once against a model.

    Args:
        cases (List[RegressionCase]): The list of regression cases
        metadata (MetaData): The metadata for the regression suite
        use_report (bool): Whether or not to use the report functionality (defaults to False)
    """

    def __init__(self, cases: List[RegressionCase], metadata: MetaData) -> None:
        self.cases: List[RegressionCase] = cases
        self.report = MultiDescriptor(name='ALL')  # TODO - allow setting names
        self.metadata = metadata
        for case in self.cases:
            self.report.parts.append(case.report)

    def get_all_subcases(self, translation: TranslationLayer) -> Iterator[Tuple[RegressionCase, str, str, str]]:
        """Get all subcases (i.e regssion case, target info and phrase) for this checker.

        Args:
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[Tuple[RegressionCase, str, str, str]]: The generator for all the cases
        """
        for case in self.cases:
            for cui, name, phrase in case.get_all_subcases(translation):
                yield case, cui, name, phrase

    def check_model(self, cat: CAT, translation: TranslationLayer,
                    total: Optional[int] = None) -> MultiDescriptor:
        """Checks model and generates a report

        Args:
            cat (CAT): The model to check against
            translation (TranslationLayer): The translation layer
            total (Optional[int]): The total number of (sub)cases expected (for a progress bar)

        Returns:
            MultiDescriptor: A report description
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
                for cui, name, phrase in case.get_all_subcases(translation):
                    if case.check_specific_for_phrase(cat, cui, name, phrase, translation):
                        successes += 1
                    else:
                        fails += 1
        return self.report

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
        d['meta'] = self.metadata.dict()
        fix_np_float64(d['meta'])

        return d

    def to_yaml(self) -> str:
        """Convert the RegressionChecker to YAML string.

        Returns:
            str: The YAML representation
        """
        return yaml.safe_dump(self.to_dict())

    def __eq__(self, other: object) -> bool:
        # only checks cases
        if not isinstance(other, RegressionChecker):
            return False
        return self.cases == other.cases

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
            if case_name == 'meta':
                continue  # ignore metadata
            case = RegressionCase.from_dict(case_name, details)
            cases.append(case)
        if 'meta' not in in_dict:
            logger.warn("Loading regression suite without any meta data")
            metadata = MetaData.unknown()
        else:
            metadata = MetaData.parse_obj(in_dict['meta'])
        return RegressionChecker(cases=cases, metadata=metadata)

    @classmethod
    def from_yaml(cls, file_name: str) -> 'RegressionChecker':
        """Constructs a RegressionChcker from a YAML file.

        The from_dict method is used for the construction from the dict.

        Args:
            file_name (str): The file name

        Returns:
            RegressionChecker: The constructed regression checker
        """
        with open(file_name) as f:
            data = yaml.safe_load(f)
        return RegressionChecker.from_dict(data)
