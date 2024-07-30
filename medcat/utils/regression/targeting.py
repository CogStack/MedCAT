import logging
from typing import Dict, Iterable, Iterator, List, Set, Tuple, Any
from functools import lru_cache
from itertools import product

from pydantic import BaseModel

from medcat.cdb import CDB

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)


class TranslationLayer:
    """The translation layer for translating:
    - CUIs to names
    - names to CUIs
    - type_ids to CUIs
    - CUIs to chil CUIs

    The idea is to decouple these translations from the CDB instance in case something changes there.

    Args:
        cui2names (Dict[str, Set[str]]): The map from CUI to names
        name2cuis (Dict[str, List[str]]): The map from name to CUIs
        cui2type_ids (Dict[str, Set[str]]): The map from CUI to type_ids
        cui2children (Dict[str, Set[str]]): The map from CUI to child CUIs
    """

    def __init__(self, cui2names: Dict[str, Set[str]], name2cuis: Dict[str, List[str]],
                 cui2type_ids: Dict[str, Set[str]], cui2children: Dict[str, Set[str]]) -> None:
        self.cui2names = cui2names
        self.name2cuis = name2cuis
        self.cui2type_ids = cui2type_ids
        self.type_id2cuis: Dict[str, Set[str]] = {}
        for cui, type_ids in self.cui2type_ids.items():
            for type_id in type_ids:
                if type_id not in self.type_id2cuis:
                    self.type_id2cuis[type_id] = set()
                self.type_id2cuis[type_id].add(cui)
        self.cui2children = cui2children
        for cui in cui2names:
            if cui not in cui2children:
                self.cui2children[cui] = set()

    def targets_for(self, cui: str) -> Iterator[Tuple[str, str]]:
        for name in self.cui2names[cui]:
            yield cui, name

    def get_first_name(self, cui: str):
        for _, name in self.targets_for(cui):
            return name

    def all_targets(self, all_cuis: List[str]) -> Iterator[Tuple[str, str]]:
        """Get a generator of all target information objects.
        This is the starting point for checking cases.

        Args:
            all_cuis (List[str]): The set of all CUIs to be queried

        Yields:
            Iterator[Tuple[str, str]]: The iterator of the target info
        """
        for cui in all_cuis:
            if cui not in self.cui2names:
                logger.warning('CUI not found in translation layer: %s', cui)
                continue
            yield from self.targets_for(cui)

    def get_direct_children(self, cui: str) -> List[str]:
        return list(self.cui2children.get(cui, []))

    @lru_cache(maxsize=10_000)
    def get_direct_parents(self, cui: str) -> List[str]:
        parents = []
        for pot_parent, children in self.cui2children.items():
            if cui in children:
                parents.append(pot_parent)
        return parents

    def get_children_of(self, found_cuis: Iterable[str], cui: str, depth: int = 1) -> List[str]:
        """Get the children of the specifeid CUI in the listed CUIs (if they exist).

        Args:
            found_cuis (Iterable[str]): The list of CUIs to look in
            cui (str): The target parent CUI
            depth (int): The depth to carry out the search for

        Returns:
            List[str]: The list of children found
        """
        if cui not in self.cui2children:
            return []  # no children
        children = self.cui2children[cui]
        found_children = []
        for child in children:
            if child in found_cuis:
                found_children.append(child)
        if depth > 1:
            for child in children:
                found_children.extend(self.get_children_of(
                    found_cuis, child, depth - 1))
        return found_children

    def get_parents_of(self, found_cuis: Iterable[str], cui: str, depth: int = 1) -> List[str]:
        """Get the parents of the specifeid CUI in the listed CUIs (if they exist).

        If needed, higher order parents (i.e grandparents) can be queries for.

        This uses the `get_children_of` method intenrnally.
        That is, if any of the found CUIs have the specified CUI as a child of
        the specified depth, the found CUIs have a parent of the specified depth.

        Args:
            found_cuis (Iterable[str]): The list of CUIs to look in
            cui (str): The target child CUI
            depth (int): The depth to carry out the search for

        Returns:
            List[str]: The list of parents found
        """
        found_parents = []
        for found_cui in found_cuis:
            if self.get_children_of({cui}, found_cui, depth=depth):
                # TODO - the intermediate results may get lost here
                # i.e if found_cui is grandparent of the specified one,
                # the direct parent is not listed
                found_parents.append(found_cui)
        return found_parents

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
        if 'pt2ch' not in cdb.addl_info:
            logger.warning(
                "No parent to child information presented so they cannot be used")
            parent2child = {}
        else:
            parent2child = cdb.addl_info['pt2ch']
        return TranslationLayer(cdb.cui2names, cdb.name2cuis, cdb.cui2type_ids, parent2child)


class FilterOptions(BaseModel):
    """A class describing the options for the filters
    """
    placeholder: str
    target_cuis: List[str]
    onlyprefnames: bool = False


class PhraseChanger(BaseModel):
    preprocess_placeholders: List[Tuple[str, str]]

    def __call__(self, phrase: str) -> str:
        for placeholder, replacement in self.preprocess_placeholders:
            if phrase.count(placeholder) != 1:
                raise # TODO - more specific error
            phrase = phrase.replace(placeholder, replacement)
        return phrase


class OptionSet(BaseModel):
    options: List[FilterOptions]
    allow_any_combinations: bool

    @classmethod
    def from_dict(cls, section: Dict[str, Any]) -> 'OptionSet':
        """Construct a FilterOptions instance from a dict.

        The assumed structure is:
        {
            'placeholders': [
                {
                'placeholder': <e.g {DIAGNOSIS}'>,
                'cui': <the CUI>,
                'prefname-only': 'true'
                }, <potentially more>],
            'any-combination': <True or False>
        }

        The prefname-only key is optional.

        Args:
            section (Dict[str, Any]): The dict to parse

        Raises:
            ValueError: If incorrect number of CUIs when not allowing any combination

        Returns:
            OptionSet: The resulting OptionSet
        """
        options: List['FilterOptions'] = []
        allow_any_in = section.get('any-combination', 'false')
        if isinstance(allow_any_in, str):
            allow_any_combinations = allow_any_in.lower() == 'true'
        elif isinstance(allow_any_in, bool):
            allow_any_combinations = allow_any_in
        else:
            raise ValueError(f"Unkown 'any-combination' value: {allow_any_in}")
        if 'placeholders' not in section:
            raise ValueError("Misconfigured - no placeholders")  # TODO - specific exception
        for part in section['placeholders']:
            placeholder = part['placeholder']
            target_cuis: List[str] = part['cuis']
            if not isinstance(target_cuis, list):
                pass # TODO - raise an exception regarding malformed config
            if 'prefname-only' in part:
                onlyprefnames = part['prefname-only'].lower() == 'true'
            else:
                onlyprefnames = False
            option = FilterOptions(placeholder=placeholder, target_cuis=target_cuis,
                                   onlyprefnames=onlyprefnames)
            options.append(option)
        if not allow_any_combinations:
            # NOTE: need to have same number of target_cuis for each placeholder
            # NOTE: there needs to be at least on option / placeholder anyway
            nr_of_cuis = [len(opt.target_cuis) for opt in options]
            if not all(nr == nr_of_cuis[0] for nr in nr_of_cuis):
                raise ValueError(f"NOT EQUAL NUMBER OF CUIS: {nr_of_cuis}")
        return OptionSet(options=options, allow_any_combinations=allow_any_combinations)

    def to_dict(self) -> dict:
        """Convert the FilterOptions to a dict.

        Returns:
            dict: The dict representation
        """
        placeholders = [
            {
                'placeholder': opt.placeholder,
                'cuis': opt.target_cuis,
                'prefname-only': str(opt.onlyprefnames),
            }
            for opt in self.options
        ]
        return {'placeholders': placeholders, 'any-combination': str(self.allow_any_combinations)}

    def _get_all_combinations(self, other_opts: List[FilterOptions],
                              translation: TranslationLayer) -> Iterator[PhraseChanger]:
        per_ph_nr_of_opts = [len(opt.target_cuis) for opt in other_opts]
        if self.allow_any_combinations:
            # for each option with N target CUIs use 0, ..., N-1
            for choosers in product(*[range(n) for n in per_ph_nr_of_opts]):
                # NOTE: using the 0th name for target CUI
                placeholders = [(opt.placeholder, translation.get_first_name(opt.target_cuis[cui_nr]))
                                for opt, cui_nr in zip(other_opts, choosers)]
                yield PhraseChanger(preprocess_placeholders=placeholders)
        else:
            # NOTE: using original options because there may not be other palceholders
            nr_of_opts = len(self.options[0].target_cuis)
            for cui_nr in range(nr_of_opts):
                placeholders = [
                    # NOTE: using the 0th name for the target CUI
                    (opt.placeholder, translation.get_first_name(opt.target_cuis[cui_nr]))
                    for opt in other_opts
                ]
                yield PhraseChanger(preprocess_placeholders=placeholders)

    def get_preprocessors_and_targets(self, translation: TranslationLayer
                                      ) -> Iterator[Tuple[PhraseChanger, str, List[str]]]:
        # TODO: based on allow_any_combination, yield ALL combinations
        #       or else yield the specified combinations
        for opt_nr in range(len(self.options)):
            other_opts = list(self.options)
            cur_opt = other_opts.pop(opt_nr)
            for changer in self._get_all_combinations(other_opts, translation):
                yield changer, cur_opt.placeholder, cur_opt.target_cuis

    def get_applicable_targets(self, translation: TranslationLayer
                               ) -> Iterator[Tuple[PhraseChanger, str, str, str]]:
        """Get all applicable targets for this filter

        Args:
            translation (TranslationLayer): The translation layer

        Yields:
            Iterator[Tuple[PhraseChanger, str, str, str]]: The output generator
        """
        for changer, placeholder, target_cuis in self.get_preprocessors_and_targets(translation):
            for cui, name in translation.all_targets(target_cuis):
                yield changer, placeholder, cui, name
