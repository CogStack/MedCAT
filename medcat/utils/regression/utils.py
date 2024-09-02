from typing import Iterator, Tuple, List, Dict, Any, Type, Callable, Set

import ast
import inspect
from enum import Enum
from functools import lru_cache
import random
import logging

from medcat.stats.mctexport import MedCATTrainerExport, MedCATTrainerExportDocument


# this placheolder will be temporarily put in the
# phrases when dealing with one that has multiple
# of the same placeholder in it
_TEMP_MULTI_PLACEHOLDER = '###===PlaceHolder===###'


def partial_substitute(phrase: str, placeholder: str, name: str, nr: int) -> str:
    """Substitute all but 1 of the many placeholders present in the given phrase.

    First, the first `nr` placeholders are replaced.
    Then the next (1) placeholder is replaced with a temporary one
    After that, the rest of the placeholders are replaced.
    And finally, the temporary placeholder is returned back to its original form.

    Example:
        If we've got `phrase = "some [PH] and [PH] we [PH]"`
        `placeholder = "[PH]"`, and `name = 'NAME'`,
        we'd get the following based on the number `nr`:
        0: "some [PH] and NAME we NAME"
        1: "some NAME and [PH] we NAME"
        2: "some NAME and NAME we [PH]"

    Args:
        phrase (str): The phrase in question.
        placeholder (str): The placeholder to replace.
        name (str): The name to replace the placeholder for.
        nr (int): The number of the target to keep.

    Raises:
        IncompatiblePhraseException: If the number of placeholders in the phrase
            is 1 or the number to be kept is too high; or the phrase has the
            temporary placeholder.

    Returns:
        str: The partially substituted phrase.
    """
    num_of_placeholder = phrase.count(placeholder)
    if nr >= num_of_placeholder or num_of_placeholder == 1:
        # NOTE: in cae of 1, this makes no sense
        raise IncompatiblePhraseException(
            f"The phrase ({repr(phrase)}) has {num_of_placeholder} "
            f"placeholders, but the {nr}th placeholder was requested to be "
            "swapped!")
    # replace stuff before the specific one
    phrase = phrase.replace(placeholder, name, nr)
    if _TEMP_MULTI_PLACEHOLDER in phrase:
        # if the temporary placeholder is already in text, the following would fail
        # unexpectedly
        raise IncompatiblePhraseException(
            f"Regression phrase with multiple placeholders ({placeholder}) "
            f"has the temporary placeholder: {repr(_TEMP_MULTI_PLACEHOLDER)}. "
            f"This means that the partial substitution of all but the {nr}th "
            "placeholder failed")
    # replace the target with temporary placeholder
    phrase = phrase.replace(placeholder, _TEMP_MULTI_PLACEHOLDER, 1)
    # replace the rest of the placeholder
    phrase = phrase.replace(placeholder, name)
    # set back the one needed placeholder
    phrase = phrase.replace(_TEMP_MULTI_PLACEHOLDER, placeholder)
    return phrase


class IncompatiblePhraseException(ValueError):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def limit_str_len(input_str: str,
                  max_length: int = 40,
                  keep_front: int = 20,
                  keep_rear: int = 10) -> str:
    """Limits the length of a string.

    If the length of the string is less than or equal to `max_length`, the same
    string is returned.
    If it's longer, the first `keep_front` are kept, then the number of chars
    is included in brackets (e.g `" [123 chars] "`), and finally the last
    `keep_rear` characters are included.

    Args:
        input_str (str): The input (potentially) long string.
        max_length (int): The maximum number of characters at which
            the string will remain unchanged. Defaults to 40.
        keep_front (int): How many starting characters to keep. Defaults to 20.
        keep_rear (int): How many ending characters to keep. Defaults to 10.

    Returns:
        str: _description_
    """
    if len(input_str) <= max_length:
        return input_str
    part1 = input_str[:keep_front]
    part2 = input_str[-keep_rear:]
    hidden_chars = len(input_str) - len(part1) - len(part2)
    return f"{part1} [{hidden_chars} chars] {part2}"


class MedCATTrainerExportConverter:
    """Used to convert an MCT export to the format required for regression."""
    # NOTE: the first placeholder will use the CUI, the 2nd the order of
    #       the annotation. This is required so that placeholders with the
    #       samme concept don't have the same name
    TEMP_PLACEHOLDER = "##[SWAPME-{}-{}]##"

    def __init__(self, mct_export: MedCATTrainerExport,
                 use_only_existing_name: bool = False) -> None:
        self.mct_export = mct_export
        self.use_only_existing_name = use_only_existing_name

    def _get_placeholder(self, cui: str, nr: int) -> str:
        return self.TEMP_PLACEHOLDER.format(cui, nr)

    def convert(self) -> dict:
        """Converts the MedCATtrainer export into regression suite dict.

        I.e this should producce a dict in the same format as one read
        from a regression suite YAML.

        Returns:
            dict: The Regression-suite compatible dict.
        """
        converted: Dict[str, dict] = {}
        for phrase, case_name, anns in self._iter_docs():
            regr_case: Dict[str, Any] = {
                'targeting': {
                    'placeholders': [
                        {
                            # NOTE: this is just and example.
                            #       it will be wiped/overwritten later
                            'placeholders': "TODO",
                            'cuis': ['CUI1']
                        }
                    ],
                    'any-combination': False,
                },
                'phrases': []  # will be filled later
            }
            placeholders: List[Dict[str, Any]] = []
            # NOTE: the iteration is done from later annotations
            #       so I can replace using the locations
            for ann_nr, (start, end, cui, _) in enumerate(anns):
                ph = self._get_placeholder(cui, ann_nr)
                phrase = phrase[:start] + ph + phrase[end:]
                placeholders.append({
                    'placeholder': ph, 'cuis': [cui, ]
                })
            # update at the very end, when changed
            regr_case['phrases'] = [phrase]
            regr_case['targeting']['placeholders'] = placeholders
            converted[case_name] = regr_case
        return converted

    def _iter_docs(self) -> Iterator[Tuple[str, str, Iterator[Tuple[int, int, str, str]]]]:
        for project in self.mct_export['projects']:
            project_id = project['id']
            project_name = project['name']
            for doc in project['documents']:
                doc_id = doc['id']
                text = doc['text']
                yield text,  f"{project_id}_{project_name}_{doc_id}", self._iter_anns_backwards(doc)

    def _iter_anns_backwards(self, doc: MedCATTrainerExportDocument) -> Iterator[Tuple[int, int, str, str]]:
        # NOTE: doing so backwards so that I can replace them one by one using the start/end,
        #       starting from the end of the phrase
        for ann in doc['annotations'][::-1]:
            yield ann['start'], ann['end'], ann['cui'], ann['value']


def get_class_level_docstrings(cls: Type) -> List[str]:
    """This is a helper method to get all the class level doc strings.

    This is designed to be used alongside and by the `add_doc_strings_to_enum` method.

    Args:
        cls (Type): The class in question.

    Returns:
        List[str]: All class-level docstrings (including the class docstring if it exists).
    """
    source_code = inspect.getsource(cls)
    tree = ast.parse(source_code)
    docstrings: List[str] = []
    # walk the tree
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for elem in node.body:
                if isinstance(elem, ast.Expr) and isinstance(elem.value, ast.Constant):
                    # If it's an expression node containing a constant, extract the string
                    docstrings.append(elem.value.s)
    return docstrings


def add_doc_strings_to_enum(cls: Type[Enum]) -> None:
    """Add doc strings to Enum as they are described in code right below each constant.

    The way python works means that the doc strings defined after an Enum constant do not
    get stored with the constant. When accessing the doc string of an Enum constant, the
    doc string of the class is returned instead.

    So what this method does is gets the doc strings by traversing the abstract syntax tree.

    While there would be easier ways to accomplish this, they would require the doc strings
    for the Enum constant to be further from the constants themselves.

    If the class itself has a doc string, it is omitted. Otherwise the Enum constants are
    given the doc strings in the order in which they appear.

    Args:
        cls (Type[Enum]): The Enum class to do this for.
    """
    docstrings = get_class_level_docstrings(cls)
    if cls.__doc__ == docstrings[0]:
        docstrings = docstrings[1:]
    for ev, ds in zip(cls, docstrings):
        ev.__doc__ = ds


@lru_cache(maxsize=10)
def get_rng(seed: int) -> random.Random:
    return random.Random(seed)


# NOTE: these are 'relatively accurate' estimates
#       that I obtained by running it on 15 different
#       concepts with names varying from length
#       of 5 to length of 74, a total of 316 names
ESTIMATION_MATRIX: Dict[int, Callable[[int], int]] = {
    1: lambda orig_len: int(52.23 * orig_len + 24.26),
    2: lambda orig_len: 2724 * orig_len**2 + 3917 * orig_len + 1098
}


def estimate_num_variants(orig_len: int, edit_distance: int) -> int:
    if edit_distance in ESTIMATION_MATRIX:
        return ESTIMATION_MATRIX[edit_distance](orig_len)
    logging.warning("Estimations for then umber of varinats for edit "
                    "distance greater than 2 (%d used) can be extremely "
                    "inaccurate.")
    # NOTE: This is a low ball estimate - the real number could be a lot bigger
    powers = list(range(0, edit_distance+1))[::-1]
    estimate_coefs = [(2 * 26) ** ed for ed in powers]
    estimated = 0
    for coef, power in zip(estimate_coefs, powers):
        estimated += coef * orig_len ** power
    return estimated


FAIL_AFTER_MULT = 10


def pick_random_edits(edit_gen: Iterator[str], num_to_pick: int,
                      orig_len: int, edit_distance: int, rng_seed: int) -> Iterator[str]:
    num_vars = estimate_num_variants(orig_len, edit_distance)
    if num_to_pick > num_vars:
        raise ValueError(f"Unable to ick {num_to_pick} out of {num_vars} "
                         f"(estimated from edit distance {edit_distance} "
                         f"and word length {orig_len})")
    rng = get_rng(rng_seed)
    pick_avoids = num_to_pick > num_vars // 2
    _num_to_pick = num_to_pick if not pick_avoids else num_vars - num_to_pick
    pick_set: Set[int] = set()
    while len(pick_set) < _num_to_pick:
        pick_set.add(rng.randint(0, num_vars))
    if pick_avoids:
        # NUMBERS NOT IN
        picks = sorted(set(range(num_vars)) - pick_set)
    else:
        picks = sorted(list(pick_set))
    for enr, edit in enumerate(edit_gen):
        if enr == picks[0]:
            picks.pop(0)
            yield edit
        if not picks:
            break
