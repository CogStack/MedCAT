from typing import Any, Dict, KeysView, Iterator, List, Tuple, Union

from medcat.cdb import CDB
from medcat.utils.saving.coding import EncodeableObject, PartEncoder, PartDecoder, UnsuitableObject, register_encoder_decoder


CUI_DICT_NAMES_TO_COMBINE = [
    "cui2names", "cui2snames", "cui2context_vectors",
    "cui2count_train", "cui2tags", "cui2type_ids",
    "cui2preferred_name", "cui2average_confidence",
]
ONE2MANY = 'cui2many'

NAME_DICT_NAMES_TO_COMBINE = [
    "cui2names", "name2cuis2status", "cui2preferred_name",
]
NAME2MANY = 'name2many'

DELEGATING_DICT_IDENTIFIER = '==DELEGATING_DICT=='


class _KeysView:
    def __init__(self, keys: KeysView, parent: 'DelegatingDict'):
        self._keys = keys
        self._parent = parent

    def __iter__(self) -> Iterator[Any]:
        for key in self._keys:
            if key in self._parent:
                yield key

    def __len__(self) -> int:
        return len([_ for _ in self])


class _ItemsView:
    def __init__(self, parent: 'DelegatingDict') -> None:
        self._parent = parent

    def __iter__(self) -> Iterator[Any]:
        for key in self._parent:
            yield key, self._parent[key]

    def __len__(self) -> int:
        return len(self._parent)


class _ValuesView:
    def __init__(self, parent: 'DelegatingDict') -> None:
        self._parent = parent

    def __iter__(self) -> Iterator[Any]:
        for key in self._parent:
            yield self._parent[key]

    def __len__(self) -> int:
        return len(self._parent)


class DelegatingDict:

    def __init__(self, delegate: Dict[str, List[Any]], nr: int,
                 nr_of_overall_items: int = 8) -> None:
        self.delegate = delegate
        self.nr = nr
        self.nr_of_overall_items = nr_of_overall_items

    def _generate_empty_entry(self) -> List[Any]:
        return [None for _ in range(self.nr_of_overall_items)]

    def __getitem__(self, key: str) -> Any:
        val = self.delegate[key][self.nr]
        if val is None:
            raise KeyError
        return val

    def get(self, key: str, default: Any) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self.delegate:
            self.delegate[key] = self._generate_empty_entry()
        self.delegate[key][self.nr] = value

    def __contains__(self, key: str) -> bool:
        return key in self.delegate and self.delegate[key][self.nr] is not None

    def keys(self) -> _KeysView:
        return _KeysView(self.delegate.keys(), self)

    def items(self) -> _ItemsView:
        return _ItemsView(self)

    def values(self) -> _ValuesView:
        return _ValuesView(self)

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def __len__(self) -> int:
        return len(self.keys())

    def to_dict(self) -> dict:
        return {'delegate': None,
                'nr': self.nr,
                'nr_of_overall_items': self.nr_of_overall_items}

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, DelegatingDict):
            return False
        return self.delegate == __value.delegate and self.nr == __value.nr

    def __hash__(self) -> int:
        return hash((self.delegate, self.nr))


class DelegatingDictEncoder(PartEncoder):

    def try_encode(self, obj):
        if isinstance(obj, DelegatingDict):
            return {DELEGATING_DICT_IDENTIFIER: obj.to_dict()}
        raise UnsuitableObject()


class DelegatingDictDecoder(PartDecoder):

    def try_decode(self, dct: dict) -> Union[dict, EncodeableObject]:
        if DELEGATING_DICT_IDENTIFIER in dct:
            info = dct[DELEGATING_DICT_IDENTIFIER]
            delegate = info['delegate']
            nr = info['nr']
            overall = info['nr_of_overall_items']
            return DelegatingDict(delegate, nr, overall)
        return dct


def attempt_fix_after_load(cdb: CDB):
    _attempt_fix_after_load(cdb, ONE2MANY, CUI_DICT_NAMES_TO_COMBINE)
    _attempt_fix_after_load(cdb, NAME2MANY, NAME_DICT_NAMES_TO_COMBINE)


# register encoder and decoders
register_encoder_decoder(encoder=DelegatingDictEncoder,
                         decoder=DelegatingDictDecoder,
                         loading_postprocessor=attempt_fix_after_load)


def _optimise(cdb: CDB, to_many_name: str, dict_names_to_combine: List[str]) -> None:
    dicts = [getattr(cdb, dict_name)
             for dict_name in dict_names_to_combine]
    one2many, delegators = map_to_many(dicts)
    for delegator, name in zip(delegators, dict_names_to_combine):
        setattr(cdb, name, delegator)
    setattr(cdb, to_many_name, one2many)
    cdb.is_dirty = True


def perform_optimisation(cdb: CDB, optimise_cuis: bool = True,
                         optimise_names: bool = False) -> None:
    """Attempts to optimise the memory footprint of the CDB.

    This can perform optimisation for cui2<...> and name2<...> dicts.
    However, by default, only cui2many optimisation will be done.
    This is because at the time of writing, there were not enough name2<...>
    dicts to be able to benefit from the optimisation.

    Does so by unifying the following dicts:

        cui2names (Dict[str, Set[str]]):
            From cui to all names assigned to it. Mainly used for subsetting (maybe even only).
        cui2snames (Dict[str, Set[str]]):
            From cui to all sub-names assigned to it. Only used for subsetting.
        cui2context_vectors (Dict[str, Dict[str, np.array]]):
            From cui to a dictionary of different kinds of context vectors. Normally you would have here
            a short and a long context vector - they are calculated separately.
        cui2count_train (Dict[str, int]):
            From CUI to the number of training examples seen.
        cui2tags (Dict[str, List[str]]):
            From CUI to a list of tags. This can be used to tag concepts for grouping of whatever.
        cui2type_ids (Dict[str, Set[str]]):
            From CUI to type id (e.g. TUI in UMLS).
        cui2preferred_name (Dict[str, str]):
            From CUI to the preferred name for this concept.
        cui2average_confidence (Dict[str, str]):
            Used for dynamic thresholding. Holds the average confidence for this CUI given the training examples.

        name2cuis (Dict[str, List[str]]):
            Map fro concept name to CUIs - one name can map to multiple CUIs.
        name2cuis2status (Dict[str, Dict[str, str]]):
            What is the status for a given name and cui pair - each name can be:
                P - Preferred, A - Automatic (e.g. let medcat decide), N - Not common.
        name2count_train (Dict[str, str]):
            Counts how often did a name appear during training.

    They will all be included in 1 dict with CUI keys and a list of values for each pre-existing dict.

    Args:
        cdb (CDB): The CDB to modify.
        optimise_cuis (bool, optional): Whether to optimise cui2<...> dicts. Defaults to True.
        optimise_names (bool, optional): Whether to optimise name2<...> dicts. Defaults to False.
    """
    # cui2<...> -> cui2many
    if optimise_cuis:
        _optimise(cdb, ONE2MANY, CUI_DICT_NAMES_TO_COMBINE)
    # name2<...> -> name2many
    if optimise_names:
        _optimise(cdb, NAME2MANY, NAME_DICT_NAMES_TO_COMBINE)


def _attempt_fix_after_load(cdb: CDB, one2many_name: str, dict_names: List[str]):
    if not hasattr(cdb, one2many_name):
        return
    one2many = getattr(cdb, one2many_name)
    for dict_name in dict_names:
        d = getattr(cdb, dict_name)
        if not isinstance(d, DelegatingDict):
            raise ValueError(f'Unknown type for {dict_name}: {type(d)}')
        d.delegate = one2many


def map_to_many(dicts: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Any]], List[DelegatingDict]]:
    one2many: Dict[str, List[Any]] = {}
    delegators: list[DelegatingDict] = []
    for nr, d in enumerate(dicts):
        delegator = DelegatingDict(
            one2many, nr, nr_of_overall_items=len(dicts))
        for key, value in d.items():
            if key not in one2many:
                one2many[key] = delegator._generate_empty_entry()
            one2many[key][nr] = value
        delegators.append(delegator)
    return one2many, delegators