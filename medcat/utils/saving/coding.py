from typing import Any, Protocol, runtime_checkable, List, Union, Type, Optional, Callable

import json


@runtime_checkable
class EncodeableObject(Protocol):

    def to_dict(self) -> dict:
        """Converts the object to a dict.

        Returns:
            dict: The dict to be serialised.
        """


class UnsuitableObject(ValueError):
    pass


class PartEncoder(Protocol):

    def try_encode(self, obj: object) -> Any:
        """Try to encode an object

        Args:
            obj (object): The object to encode

        Raises:
            UnsuitableObject: If the object is unsuitable for encoding.

        Returns:
            Any: The encoded object
        """


SET_IDENTIFIER = '==SET=='


class SetEncoder(PartEncoder):
    """JSONEncoder (and decoder) for sets.

    Generally, JSON doesn't support serializing of sets natively.
    This encoder adds a set identifier to the data when being serialized
    and provides a method to read said identifier upon decoding."""

    def try_encode(self, obj):
        if isinstance(obj, set):
            return {SET_IDENTIFIER: list(obj)}
        raise UnsuitableObject()


class PartDecoder(Protocol):

    def try_decode(self, dct: dict) -> Union[dict, Any]:
        """Try to decode the dictionary.

        Args:
            dct (dict): The dict to decode.

        Returns:
            Union[dict, Any]: The dict if unable to decode, the decoded object otherwise
        """


class SetDecoder(PartDecoder):

    def try_decode(self, dct: dict) -> Union[dict, set]:
        """Decode sets from input dicts.

        Args:
            dct (dict): The input dict

        Returns:
            Union[dict, set]: The original dict if this was not a serialized set, the set otherwise
        """
        if SET_IDENTIFIER in dct:
            return set(dct[SET_IDENTIFIER])
        return dct


PostProcessor = Callable[[Any], None]  # CDB -> None

DEFAULT_ENCODERS: List[Type[PartEncoder]] = [SetEncoder, ]
DEFAULT_DECODERS: List[Type[PartDecoder]] = [SetDecoder, ]
LOADING_POSTPROCESSORS: List[PostProcessor] = []


def register_encoder_decoder(encoder: Optional[Type[PartEncoder]],
                             decoder: Optional[Type[PartDecoder]],
                             loading_postprocessor: Optional[PostProcessor]):
    if encoder:
        DEFAULT_ENCODERS.append(encoder)
    if decoder:
        DEFAULT_DECODERS.append(decoder)
    if loading_postprocessor:
        LOADING_POSTPROCESSORS.append(loading_postprocessor)


class CustomDelegatingEncoder(json.JSONEncoder):

    def __init__(self, delegates: List[PartEncoder], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._delegates = delegates

    def default(self, obj):
        for delegator in self._delegates:
            try:
                return delegator.try_encode(obj)
            except UnsuitableObject:
                pass
        return json.JSONEncoder.default(self, obj)

    @classmethod
    def def_inst(cls, *args, **kwargs) -> 'CustomDelegatingEncoder':
        return cls([_cls() for _cls in DEFAULT_ENCODERS], *args, **kwargs)


class CustomDelegatingDecoder(json.JSONDecoder):
    _def_inst: Optional['CustomDelegatingDecoder'] = None

    def __init__(self, delegates: List[PartDecoder]) -> None:
        self._delegates = delegates

    def object_hook(self, dct: dict) -> Any:
        for delegator in self._delegates:
            ret_val = delegator.try_decode(dct)
            if ret_val is not dct:
                return ret_val
        return dct

    @classmethod
    def def_inst(cls) -> 'CustomDelegatingDecoder':
        if cls._def_inst is None:
            cls._def_inst = cls([_cls() for _cls in DEFAULT_DECODERS])
        return cls._def_inst


def default_hook(dct: dict) -> Any:
    cdd = CustomDelegatingDecoder.def_inst()
    return cdd.object_hook(dct)


def default_postprocessing(cdb) -> None:
    for pp in LOADING_POSTPROCESSORS:
        pp(cdb)
