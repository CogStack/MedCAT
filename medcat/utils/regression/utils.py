from typing import Iterator, Tuple, List, Dict, Any

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

    If the lenght of the string is less than or equal to `max_length`, the same
    string is returned.
    If it's longer, the first `keep_front` are kept, then the number of chars
    is included in brackets (e.g `" [123 chars] "`), and finally the last
    `keeo_rear` characters are included.

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
