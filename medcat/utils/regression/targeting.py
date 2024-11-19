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
                 cui2type_ids: Dict[str, Set[str]], cui2children: Dict[str, Set[str]],
                 cui2preferred_names: Dict[str, str],
                 separator: str, whitespace: str = ' ') -> None:
        self.cui2names = cui2names
        self.name2cuis = name2cuis
        self.cui2type_ids = cui2type_ids
        self.cui2preferred_names = cui2preferred_names
        self.separator = separator
        self.whitespace = whitespace
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

    def get_names_of(self, cui: str, only_prefnames: bool) -> List[str]:
        """Get the preprocessed names of a CUI.

        This method preporcesses the names by replacing the separator (generally `~`)
        with the appropriate whitespace (` `).

        If the concept is not in the underlying CDB, an empty list is returned.

        Args:
            cui (str): The concept in question.
            only_prefnames (bool): Whether to only return a preferred name.

        Returns:
            List[str]: The list of names.
        """
        if only_prefnames:
            return [self.get_preferred_name(cui).replace(self.separator, self.whitespace)]
        return [name.replace(self.separator, self.whitespace)
                   # NOTE: sorting the order here in case we're using
                   #       edirts in which case the order of the names
                   #       needs to be the same, otherwise different
                   #       edits will be used across runs
                   for name in sorted(self.cui2names.get(cui, []))]

    def get_preferred_name(self, cui: str) -> str:
        """Get the preferred name of a concept.

        If no preferred name is found, the random 'first' name is selected.

        Args:
            cui (str): The concept ID.

        Returns:
            str: The preferred name.
        """
        pref_name = self.cui2preferred_names.get(cui, None)
        if pref_name is None:
            logger.warning("CUI %s does not have a preferred name. "
                           "Using a random 'first' name of all the names", cui)
            return self.get_first_name(cui)
        return pref_name

    def get_first_name(self, cui: str) -> str:
        """Get the preprocessed (potentially) arbitrarily first name of the given concept.

        If the concept does not exist, the CUI itself is returned.

        PS: The "first" name may not be consistent across runs since it relies on set order.

        Args:
            cui (str): The concept ID.

        Returns:
            str: The first name.
        """
        for name in self.cui2names.get(cui, [cui]):
            return name.replace(self.separator, self.whitespace)
        return cui

    def get_direct_children(self, cui: str) -> List[str]:
        """Get the direct children of a concept.

        This means only the children, but not grandchildren.

        If the underlying CDB doesn't list children for this CUI, an empty list is returned.

        Args:
            cui (str): The concept in question.

        Returns:
            List[str]: The (potentially empty) list of direct children.
        """
        return list(self.cui2children.get(cui, []))

    @lru_cache(maxsize=10_000)
    def get_direct_parents(self, cui: str) -> List[str]:
        """Get the direct parent(s) of a concept.

        PS: This method can be quite a CPU heavy one since it relies
            on running through all the parent-children relationships
            since the child->parent(s) relationship isn't normally
            kept track of.

        Args:
            cui (str): _description_

        Returns:
            List[str]: _description_
        """
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
        return TranslationLayer(cdb.cui2names, cdb.name2cuis, cdb.cui2type_ids, parent2child,
                                cui2preferred_names=cdb.cui2preferred_name,
                                separator=cdb.config.general.separator)


class TargetPlaceholder(BaseModel):
    """A class describing the options for a specific placeholder.
    """
    placeholder: str
    target_cuis: List[str]
    onlyprefnames: bool = False


class PhraseChanger(BaseModel):
    """The phrase changer.

    This is class used as a preprocessor for phrases with multiple placeholders.
    It allows swapping in the rest of the placeholders while leaving in the one
    that's being tested for.
    """
    preprocess_placeholders: List[Tuple[str, str]]

    def __call__(self, phrase: str) -> str:
        for placeholder, replacement in self.preprocess_placeholders:
            phrase = phrase.replace(placeholder, replacement)
        return phrase

    @classmethod
    def empty(cls) -> 'PhraseChanger':
        """Gets the empty phrase changer.

        That is a phrase changer that makes no changes to the phrase.

        Returns:
            PhraseChanger: The empty phrase changer.
        """
        return cls(preprocess_placeholders=[])


class TargetedPhraseChanger(BaseModel):
    """The target phrase changer.

    It includes the phrase changer (for preprocessing) along with
    the relevant concept and the placeholder it will replace.
    """
    changer: PhraseChanger
    placeholder: str
    cui: str
    onlyprefnames: bool


class FinalTarget(BaseModel):
    """The final target.

    This involves the final phrase (which (potentially) has other
    placeholder replaced in it), the placeholder to be replaced,
    and the CUI and specific name being used.
    """
    placeholder: str
    cui: str
    name: str
    final_phrase: str


class OptionSet(BaseModel):
    """The targeting option set.

    This describes all the target placeholders and concepts needed.
    """
    options: List[TargetPlaceholder]
    allow_any_combinations: bool = False

    @classmethod
    def from_dict(cls, section: Dict[str, Any]) -> 'OptionSet':
        """Construct a OptionSet instance from a dict.

        The assumed structure is:
        {
            'placeholders': [
                {
                'placeholder': <e.g {DIAGNOSIS}'>,
                'cuis': <the CUI>,
                'prefname-only': 'true'
                }, <potentially more>],
            'any-combination': <True or False>
        }

        The prefname-only key is optional.

        Args:
            section (Dict[str, Any]): The dict to parse

        Raises:
            ProblematicOptionSetException: If incorrect number of CUIs when not allowing any combination
            ProblematicOptionSetException: If placeholders not a list
            ProblematicOptionSetException: If multiple placehodlers with same place holder

        Returns:
            OptionSet: The resulting OptionSet
        """
        options: List['TargetPlaceholder'] = []
        allow_any_in = section.get('any-combination', 'false')
        if isinstance(allow_any_in, str):
            allow_any_combinations = allow_any_in.lower() == 'true'
        elif isinstance(allow_any_in, bool):
            allow_any_combinations = allow_any_in
        else:
            raise ProblematicOptionSetException(f"Unknown 'any-combination' value: {allow_any_in}")
        if 'placeholders' not in section:
            raise ProblematicOptionSetException("Misconfigured - no placeholders")
        section_placeholders = section['placeholders']
        if not isinstance(section_placeholders, list):
            raise ProblematicOptionSetException("Misconfigured - placehodlers not a list "
                                                f"({section_placeholders})")
        used_ph = set()
        for part in section_placeholders:
            placeholder = part['placeholder']
            if not isinstance(placeholder, str):
                raise ProblematicOptionSetException(f"Unknown placeholder of type {type(placeholder)}. "
                                                    "Expected a string. Perhaps you need to surrong the "
                                                    "placeholder with single quotes (') in the yaml? "
                                                    f"Received: {placeholder}")
            if placeholder in used_ph:
                raise ProblematicOptionSetException("Misconfigured - multiple identical placeholders")
            used_ph.add(placeholder)
            target_cuis: List[str] = part['cuis']
            if not isinstance(target_cuis, list):
                raise ProblematicOptionSetException(
                    f"Target CUIs not a list ({type(target_cuis)}): {repr(target_cuis)}")
            if 'prefname-only' in part:
                opn = part['prefname-only']
                if isinstance(opn, bool):
                    onlyprefnames = opn
                else:
                    onlyprefnames = str(opn).lower() == 'true'
            else:
                onlyprefnames = False
            option = TargetPlaceholder(placeholder=placeholder, target_cuis=target_cuis,
                                   onlyprefnames=onlyprefnames)
            options.append(option)
        if not options:
            raise ProblematicOptionSetException("Misconfigured - 0 placeholders found (empty list)")
        if not allow_any_combinations:
            # NOTE: need to have same number of target_cuis for each placeholder
            # NOTE: there needs to be at least on option / placeholder anyway
            nr_of_cuis = [len(opt.target_cuis) for opt in options]
            if not all(nr == nr_of_cuis[0] for nr in nr_of_cuis):
                raise ProblematicOptionSetException(
                    f"Unequal number of cuis when any-combination: false: {nr_of_cuis}. "
                    "When any-combination: false the number of CUIs for each placeholder "
                    "should be equal.")
        return OptionSet(options=options, allow_any_combinations=allow_any_combinations)

    def to_dict(self) -> dict:
        """Convert the OptionSet to a dict.

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

    def _get_all_combinations(self, cur_opts: TargetPlaceholder, other_opts: List[TargetPlaceholder],
                              translation: TranslationLayer) -> Iterator[Tuple[PhraseChanger, str]]:
        per_ph_nr_of_opts = [len(opt.target_cuis) for opt in other_opts]
        if self.allow_any_combinations:
            # for each option with N target CUIs use 0, ..., N-1
            for choosers in product(*[range(n) for n in per_ph_nr_of_opts]):
                # NOTE: using the 0th name for target CUI
                placeholders = [(opt.placeholder, translation.get_preferred_name(opt.target_cuis[cui_nr]))
                                for opt, cui_nr in zip(other_opts, choosers)]
                for target_cui in cur_opts.target_cuis:
                    yield PhraseChanger(preprocess_placeholders=placeholders), target_cui
        else:
            nr_of_opts = len(cur_opts.target_cuis)
            for cui_nr in range(nr_of_opts):
                placeholders = [
                    # NOTE: using the 0th name for the target CUI
                    (opt.placeholder, translation.get_preferred_name(opt.target_cuis[cui_nr]))
                    for opt in other_opts
                ]
                yield PhraseChanger(preprocess_placeholders=placeholders), cur_opts.target_cuis[cui_nr]

    def estimate_num_of_subcases(self) -> int:
        """Get the number of distinct subcases.

        This includes ones that can be calculated without the knowledge of the
        underlying CDB. I.e it doesn't care for the number of names involved per CUI
        but only takes into account what is described in the option set itself.

        If any combination is allowed, then the answer is the combination of
        the number of target concepts per option.
        If any combination is not allowed, then the answer is simply the number
        of target concepts for an option (they should all have the same number).

        Returns:
            int: _description_
        """
        num_of_opts = len(self.options)
        if self.allow_any_combinations:
            total_cases = 1
            for cur_opt in self.options:
                total_cases *= len(cur_opt.target_cuis)
        else:
            total_cases = len(self.options[0].target_cuis)
        return num_of_opts * total_cases

    def get_preprocessors_and_targets(self, translation: TranslationLayer
                                      ) -> Iterator[TargetedPhraseChanger]:
        """Get the targeted phrase changers.

        Args:
            translation (TranslationLayer): The translaton layer.

        Yields:
            Iterator[TargetedPhraseChanger]: Thetarget phrase changers.
        """
        num_of_opts = len(self.options)
        if num_of_opts == 1:
            # NOTE: when there's only 1 option, the other option doesn't work
            #       since it has nothing to iterate over regarding 'other' options
            opt = self.options[0]
            for target_cui in opt.target_cuis:
                yield TargetedPhraseChanger(changer=PhraseChanger.empty(),
                                            placeholder=opt.placeholder,
                                            cui=target_cui,
                                            onlyprefnames=opt.onlyprefnames)
            return
        for opt_nr in range(num_of_opts):
            other_opts = list(self.options)
            cur_opt = other_opts.pop(opt_nr)
            for changer, target_cui in self._get_all_combinations(cur_opt, other_opts, translation):
                yield TargetedPhraseChanger(changer=changer,
                                            placeholder=cur_opt.placeholder,
                                            cui=target_cui,
                                            onlyprefnames=cur_opt.onlyprefnames)


class ProblematicOptionSetException(ValueError):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
