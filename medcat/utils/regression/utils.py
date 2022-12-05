from enum import Enum
from typing import Type, TypeVar


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
