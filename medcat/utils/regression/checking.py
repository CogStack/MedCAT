from typing import Any, Dict, Iterator, List, Tuple, Optional
import yaml
import json
import logging
import tqdm
import datetime
import os

from pydantic import BaseModel, Field

from medcat.cat import CAT
from medcat.utils.regression.targeting import TranslationLayer, OptionSet
from medcat.utils.regression.targeting import FinalTarget, TargetedPhraseChanger
from medcat.utils.regression.utils import partial_substitute, MedCATTrainerExportConverter
from medcat.utils.regression.utils import pick_random_edits
from medcat.utils.regression.results import MultiDescriptor, ResultDescriptor, Finding
from medcat.utils.normalizers import get_all_edits_n

logger = logging.getLogger(__name__)


class RegressionCase(BaseModel):
    """A regression case that has a name, defines options, filters and phrases.
    """
    name: str
    options: OptionSet
    phrases: List[str]
    report: ResultDescriptor

    def check_specific_for_phrase(self, cat: CAT, target: FinalTarget,
                                  translation: TranslationLayer) -> Tuple[Finding, Optional[str]]:
        """Checks whether the specific target along with the specified phrase
        is able to be identified using the specified model.

        Args:
            cat (CAT): The model
            target (FinalTarget): The final target configuration
            translation (TranslationLayer): The translation layer

        Raises:
            MalformedRegressionCaseException: If there are too many placeholders in phrase.

        Returns:
            Tuple[Finding, Optional[str]]: The nature to which the target was (or wasn't) identified
        """
        phrase, cui, name, placeholder = target.final_phrase, target.cui, target.name, target.placeholder
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
        self.report.report(target, finding)
        return finding

    def estimate_num_of_diff_subcases(self) -> int:
        return len(self.phrases) * self.options.estimate_num_of_subcases()

    def get_distinct_cases(self, translation: TranslationLayer,
                           edit_distance: Tuple[int, int, int],
                           use_diacritics: bool) -> Iterator[Iterator[FinalTarget]]:
        """Gets the various distinct sub-case iterators.

        The sub-cases are those that can be determine without the translation layer.
        However, the translation layer is included here since it streamlines the operation.

        Args:
            translation (TranslationLayer): The translation layer.
            edit_distance (Tuple[int, int, int]): The edit distance(s) to try.
            use_diacritics (bool): Whether to use diacritics for edit distance.

        Yields:
            Iterator[Iterator[FinalTarget]]: The iterator of iterators of different sub cases.
        """
        # for each phrase and for each placeholder based option
        for changer in self.options.get_preprocessors_and_targets(translation):
            for phrase in self.phrases:
                yield self._get_subcases(phrase, changer, translation, edit_distance, use_diacritics)

    def _get_subcases(self, phrase: str, changer: TargetedPhraseChanger,
                      translation: TranslationLayer,
                      edit_distance: Tuple[int, int, int],
                      use_diacritics: bool,
                      ) -> Iterator[FinalTarget]:
        cui, placeholder = changer.cui, changer.placeholder
        changed_phrase = changer.changer(phrase)
        edit_dist, edit_rn_seed, edit_pick = edit_distance
        for raw_name in translation.get_names_of(cui, changer.onlyprefnames):
            name_variant = 0
            if edit_dist:# TODO: use config.ner.min_name_len or something
                name_gen = get_all_edits_n(
                    raw_name, use_diacritics, edit_dist, return_ordered=True)
                all_names = list(pick_random_edits(name_gen, edit_pick, len(raw_name),
                                                   edit_dist, edit_rn_seed))
            else:
                all_names = [raw_name]
            for name in all_names:
                if edit_dist:
                    logger.debug("Changed name from '%s' to '%s' (variant %d, edit distance %s, "
                                 "seed %d, picking %d)",
                                 raw_name, name, name_variant, edit_dist,
                                 edit_rn_seed, edit_pick)
                    name_variant += 1
                num_of_phs = changed_phrase.count(placeholder)
                if num_of_phs == 1:
                    yield FinalTarget(placeholder=placeholder,
                                    cui=cui, name=name,
                                    final_phrase=changed_phrase)
                    continue
                for cntr in range(num_of_phs):
                    final_phrase = partial_substitute(changed_phrase, placeholder, name, cntr)
                    yield FinalTarget(placeholder=placeholder,
                                    cui=cui, name=name,
                                    final_phrase=final_phrase)

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

        The expected structure:
        {
            'targeting': {
                [
                    'placeholder': '[DIAGNOSIS]'  # the placeholder to be replaced
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

    The ontology is assumed to be described at:
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
    """The metadata for the regression suite.

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
    """Fix numpy.float64 in dictionary for yaml saving purposes.

    These types of objects are unable to be cleanly serialized using yaml.
    So we need to convert them to the corresponding floats.

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


class RegressionSuite:
    """The regression checker.
    This is used to check a bunch of regression cases at once against a model.

    Args:
        cases (List[RegressionCase]): The list of regression cases
        metadata (MetaData): The metadata for the regression suite
        use_report (bool): Whether or not to use the report functionality (defaults to False)
    """

    def __init__(self, cases: List[RegressionCase], metadata: MetaData, name: str) -> None:
        self.cases: List[RegressionCase] = cases
        self.report = MultiDescriptor(name=name)
        self.metadata = metadata
        for case in self.cases:
            self.report.parts.append(case.report)

    def get_all_distinct_cases(self, translation: TranslationLayer,
                               edit_distance: Tuple[int, int, int],
                               use_diacritics: bool
                               ) -> Iterator[Tuple[RegressionCase, Iterator[FinalTarget]]]:
        """Gets all the distinct cases for this regression suite.

        While distinct cases can be determined without the translation layer,
        including it here simplifies the process.

        Args:
            translation (TranslationLayer): The translation layer.
            edit_distance (Tuple[int, int, int]): The edit distance(s) to try.
                Defaults to (0, 0, 0).
            use_diacritics (bool): Whether to use diacritics for edit distance.

        Yields:
            Iterator[Tuple[RegressionCase, Iterator[FinalTarget]]]: The generator of the
                regression case along with its corresponding sub-cases.
        """
        for regr_case in self.cases:
            for subcase in regr_case.get_distinct_cases(translation, edit_distance,
                                                        use_diacritics):
                yield regr_case, subcase

    def estimate_total_distinct_cases(self) -> int:
        return sum(rc.estimate_num_of_diff_subcases() for rc in self.cases)

    def iter_subcases(self, translation: TranslationLayer,
                      show_progress: bool = True,
                      edit_distance: Tuple[int, int, int] = (0, 0, 0),
                      use_diacritics: bool = False,
                      ) -> Iterator[Tuple[RegressionCase, FinalTarget]]:
        """Iterate over all the sub-cases.

        Each sub-case present a unique target (phrase, concept, name) on
        the corresponding regression case.

        Args:
            translation (TranslationLayer): The translation layer.
            show_progress (bool): Whether to show progress. Defaults to True.
            edit_distance (Tuple[int, int, int]): The edit distance(s) to try.
                Defaults to (0, 0, 0).
            use_diacritics (bool): Whether to use diacritics for edit distance.

        Yields:
            Iterator[Tuple[RegressionCase, FinalTarget]]: The generator of the
                regression case along with each of the final target sub-cases.
        """
        total = self.estimate_total_distinct_cases()
        for (regr_case, subcase) in tqdm.tqdm(self.get_all_distinct_cases(translation,
                                                                          edit_distance,
                                                                          use_diacritics),
                                              total=total, disable=not show_progress):
            for target in subcase:
                yield regr_case, target

    def check_model(self, cat: CAT, translation: TranslationLayer,
                    edit_distance: Tuple[int, int, int] = (0, 0, 0),
                    use_diacritics: bool = False,
                    ) -> MultiDescriptor:
        """Checks model and generates a report

        Args:
            cat (CAT): The model to check against
            translation (TranslationLayer): The translation layer
            edit_distance (Tuple[int, int, int]): The edit distance of the names.
                Defaults to (0, 0, 0).
            use_diacritics (bool): Whether to use diacritics for edit distance.

        Returns:
            MultiDescriptor: A report description
        """
        for regr_case, target in self.iter_subcases(translation, True,
                                                    edit_distance, use_diacritics):
            # NOTE: the finding is reported in the per-case report
            regr_case.check_specific_for_phrase(cat, target, translation)
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
        d['meta'] = self.metadata.model_dump()
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
        if not isinstance(other, RegressionSuite):
            return False
        return self.cases == other.cases

    @classmethod
    def from_dict(cls, in_dict: dict, name: str) -> 'RegressionSuite':
        """Construct a RegressionChecker from a dict.

        Most of the parsing is handled in RegressionChecker.from_dict.
        This just assumes that each key in the dict is a name
        and each value describes a RegressionCase.

        Args:
            in_dict (dict): The input dict.
            name (str): The name of the regression suite.

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
        return RegressionSuite(cases=cases, metadata=metadata, name=name)

    @classmethod
    def from_yaml(cls, file_name: str) -> 'RegressionSuite':
        """Constructs a RegressionChcker from a YAML file.

        The from_dict method is used for the construction from the dict.

        Args:
            file_name (str): The file name

        Returns:
            RegressionChecker: The constructed regression checker
        """
        with open(file_name) as f:
            data = yaml.safe_load(f)
        return RegressionSuite.from_dict(data, name=os.path.basename(file_name))

    @classmethod
    def from_mct_export(cls, file_name: str) -> 'RegressionSuite':
        with open(file_name) as f:
            data = json.load(f)
        converted = MedCATTrainerExportConverter(data).convert()
        return RegressionSuite.from_dict(converted, name=os.path.basename(file_name))


class MalformedRegressionCaseException(ValueError):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
