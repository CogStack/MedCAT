
# this placheolder will be temporarily put in the
# phrases when dealing with one that has multiple
# of the same placeholder in it
_TEMP_MULTI_PLACEHOLDER = '###===PlaceHolder===###'


def partial_substitute(phrase: str, placeholder: str, name: str, nr: int) -> str:
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
    if len(input_str) <= max_length:
        return input_str
    part1 = input_str[:keep_front]
    part2 = input_str[-keep_rear:]
    hidden_chars = len(input_str) - len(part1) - len(part2)
    return f"{part1} [{hidden_chars} chars] {part2}"
