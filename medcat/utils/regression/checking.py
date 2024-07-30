from typing import Any, Dict, Iterator, List, Optional, Tuple
import yaml
import logging
import tqdm
import datetime

from pydantic import BaseModel, Field

from medcat.cat import CAT
from medcat.utils.regression.targeting import TranslationLayer, OptionSet, PhraseChanger
from medcat.utils.regression.utils import partial_substitute
from medcat.utils.regression.results import MultiDescriptor, ResultDescriptor, Finding

logger = logging.getLogger(__name__)


class RegressionCase(BaseModel):
    """A regression case that has a name, defines options, filters and phrases.
    """
    name: str
    options: OptionSet
    phrases: List[str]
    report: ResultDescriptor

    def get_all_targets(self, translation: TranslationLayer
                        ) -> Iterator[Tuple[PhraseChanger, str, str, str]]:
        """Get all applicable targets for this regression case

        Args:
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[Tuple[PhraseChanger, str, str, str]]: The output generator
        """
        yield from self.options.get_applicable_targets(translation)

    def check_specific_for_phrase(self, cat: CAT, cui: str, name: str, phrase: str,
                                  translation: TranslationLayer,
                                  placeholder: str = '%s') -> Finding:
        """Checks whether the specific target along with the specified phrase
        is able to be identified using the specified model.

        Args:
            cat (CAT): The model
            cui (str): The target CUI
            name (str): The target name
            phrase (str): The phrase to check
            translation (TranslationLayer): The translation layer
            placeholder (str): The placeholder to replace. Defaults to '%s'.

        Raises:
            MalformedRegressionCaseException: If there are too many placeholders in phrase.

        Returns:
            Finding: The nature to which the target was (or wasn't) identified
        """
        nr_of_placeholders = phrase.count(placeholder)
        if nr_of_placeholders != 1:
            raise MalformedRegressionCaseException(f"Got {nr_of_placeholders} placeholders "
                                                   f"({placeholder}) (expected 1) for phrase: " +
                                                   phrase)
        ph_start = phrase.find(placeholder)
        res = cat.get_entities(phrase.replace(placeholder, name), only_cui=False)
        ents = res['entities']
        finding = Finding.determine(cui, ph_start, ph_start + len(name),
                                    translation, ents)
        if finding is Finding.IDENTICAL:
            logger.debug(
                'Matched test case %s in phrase "%s"', (cui, name), phrase)
        else:
            found_cuis = [ents[nr]['cui'] for nr in ents]
            found_names = [ents[nr]['source_value'] for nr in ents]
            cuis_names = ', '.join([f'{fcui}|{fname}'
                                    for fcui, fname in zip(found_cuis, found_names)])
            logger.debug(
                'FAILED to (fully) match (%s) test case %s in phrase "%s", '
                'found the following CUIS/names: %s', finding, (cui, name), phrase, cuis_names)
        self.report.report(cui, name, phrase, finding)
        return finding

    def get_all_subcases(self, translation: TranslationLayer) -> Iterator[Tuple[str, str, str, str]]:
        """Get all subcases for this case.
        That is, all combinations of targets with their appropriate phrases.

        Args:
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[Tuple[str, str, str, str]]: The generator for the target info and the phrase
        """
        for changer, placeholder, cui, name in self.get_all_targets(translation):
            for phrase in self.phrases:
                # NOTE: yielding the prhase as changed by the additional / other placeholders
                changed_phrase = changer(phrase)
                num_of_phs = changed_phrase.count(placeholder)
                if num_of_phs == 1:
                    yield placeholder, cui, name, changed_phrase
                    return
                for cntr in range(num_of_phs):
                    final_phrase = partial_substitute(changed_phrase, placeholder, name, cntr)
                    yield placeholder, cui, name, final_phrase

    def check_case(self, cat: CAT, translation: TranslationLayer) -> Dict[Finding, int]:
        """Check the regression case against a model.
        I.e check all its applicable targets.

        Args:
            cat (CAT): The CAT instance
            translation (TranslationLayer): The translation layer

        Returns:
            Dict[Finding, int]: The total findings.
        """
        findings: Dict[Finding, int] = {}
        for placeholder, cui, name, phrase in self.get_all_subcases(translation):
            finding = self.check_specific_for_phrase(cat, cui, name, phrase, translation,
                                                     placeholder=placeholder)
            if finding not in findings:
                findings[finding] = 0
            findings[finding] += 1
        return findings

    def to_dict(self) -> dict:
        """Converts the RegressionCase to a dict for serialisation.

        Returns:
            dict: The dict representation
        """
        d: Dict[str, Any] = {'phrases': list(self.phrases)}
        targeting = self.options.to_dict()
        d['targeting'] = targeting
        return d

    @classmethod
    def from_dict(cls, name: str, in_dict: dict) -> 'RegressionCase':
        """Construct the regression case from a dict.

        The expected stucture:
        {
            'targeting': {
                [
                    'placeholder': '[DIAGNOSIS]'  # the placeholder to be repalced
                    'cuis': ['cui1', 'cui2']
                    'prefname-only': 'false', # optional
                ]
            },
            'phrases': ['phrase %s'] # possible multiple
        }

        Args:
            name (str): The name of the case
            in_dict (dict): The dict describing the case

        Raises:
            ValueError: If the input dict does not have the 'targeting' section
            ValueError: If there are no phrases defined

        Returns:
            RegressionCase: The constructed regression cases.
        """
        # set up targeting
        if 'targeting' not in in_dict:
            raise ValueError('Input dict should define targeting')
        targeting_section = in_dict['targeting']
        # set up options
        options = OptionSet.from_dict(targeting_section)
        # all_cases: List['RegressionCase'] = []
        # for option in options:
        #     # set up test phrases
        if 'phrases' not in in_dict:
            raise ValueError('Input dict should defined phrases')
        phrases = in_dict['phrases']
        if not isinstance(phrases, list):
            phrases = [phrases]  # just one defined
        if not phrases:
            raise ValueError('Need at least one target phrase')
        return RegressionCase(name=name, options=options,
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

    def get_all_subcases(self, translation: TranslationLayer) -> Iterator[Tuple[RegressionCase, str, str, str, str]]:
        """Get all subcases (i.e regssion case, target info and phrase) for this checker.

        Args:
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[Tuple[RegressionCase, str, str, str]]: The generator for all the cases
        """
        for case in self.cases:
            for placeholder, cui, name, phrase in case.get_all_subcases(translation):
                yield case, placeholder, cui, name, phrase

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
        if total is not None:
            for regr_case, placeholder, ti, phrase in tqdm.tqdm(self.get_all_subcases(translation), total=total):
                # NOTE: the finding is reported in the per-case report
                regr_case.check_specific_for_phrase(cat, ti, phrase, translation, placeholder)
        else:
            for regr_case in tqdm.tqdm(self.cases):
                for placeholder, cui, name, phrase in regr_case.get_all_subcases(translation):
                    # NOTE: the finding is reported in the per-case report
                    regr_case.check_specific_for_phrase(cat, cui, name, phrase, translation, placeholder)
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
        cases: List[RegressionCase] = []
        for case_name, details in in_dict.items():
            if case_name == 'meta':
                continue  # ignore metadata
            add_case = RegressionCase.from_dict(case_name, details)
            cases.append(add_case)
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


class MalformedRegressionCaseException(ValueError):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
