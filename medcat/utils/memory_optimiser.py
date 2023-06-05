from typing import Any, Dict, KeysView, ValuesView, ItemsView, Iterator, List, Tuple

from medcat.cdb import CDB


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

    def keys(self) -> Iterator[str]:
        return self.KeysView(self.delegate.keys(), self)

    def items(self) -> Iterator[Tuple[str, Any]]:
        return self.ItemsView(self)

    def values(self) -> Iterator[Any]:
        return self.ValuesView(self)

    def __iter__(self) -> Iterator[str]:
        yield from self.keys()

    def __len__(self) -> int:
        return len(self.keys())

    class KeysView:
        def __init__(self, keys: KeysView, parent: 'DelegatingDict'):
            self._keys = keys
            self._parent = parent

        def __iter__(self) -> Iterator[Any]:
            for key in self._keys:
                if key in self._parent:
                    yield key

        def __len__(self) -> int:
            return len([_ for _ in self])

    class ItemsView:
        def __init__(self, parent: 'DelegatingDict') -> None:
            self._parent = parent

        def __iter__(self) -> Iterator[Any]:
            for key in self._parent:
                yield key, self._parent[key]

        def __len__(self) -> int:
            return len(self._parent)

    class ValuesView:
        def __init__(self, parent: 'DelegatingDict') -> None:
            self._parent = parent

        def __iter__(self) -> Iterator[Any]:
            for key in self._parent:
                yield self._parent[key]

        def __len__(self) -> int:
            return len(self._parent)


def _optimise(cdb: CDB, to_many_name: str, dict_names_to_combine: List[str]) -> None:
    dicts = [getattr(cdb, dict_name)
             for dict_name in dict_names_to_combine]
    one2many, delegators = map_to_many(dicts)
    for delegator, name in zip(delegators, dict_names_to_combine):
        setattr(cdb, name, delegator)
    setattr(cdb, to_many_name, one2many)
    cdb.is_dirty = True


def perform_optimisation(cdb: CDB) -> None:
    """Attempts to optimise the memory footprint of the CDB.

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
    """
    # cui2<...> -> cui2many
    cui_dict_names_to_combine = [
        "cui2names", "cui2snames", "cui2context_vectors",
        "cui2count_train", "cui2tags", "cui2type_ids",
        "cui2preferred_name", "cui2average_confidence",
    ]
    _optimise(cdb, 'cui2many', cui_dict_names_to_combine)
    # name2<...> -> name2many
    name_dict_names_to_combine = [
        "cui2names", "name2cuis2status", "cui2preferred_name",
    ]
    _optimise(cdb, 'name2many', name_dict_names_to_combine)


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


# TODO - remove anything below
def main(file_name: str):
    import dill
    d1 = {'c1': ['n11', 'n12'],
          'c2': ['n21', 'n22']}
    d2 = {'c1': 'n11',
          'c2': 'n22'}
    one2many, (delegate1, delegate2) = map_to_many([d1, d2])
    print('DEL1, DEL2', delegate1, delegate2)
    print('O2M ', one2many)
    print('DEL1', delegate1.delegate)
    print('DEL2', delegate2.delegate)
    print('COMP1', delegate1.delegate is one2many)
    print('COMP2', delegate2.delegate is one2many)
    print('COMP3', delegate1.delegate is delegate2.delegate)
    to_save = {'one2many': one2many,
               'delegate1': delegate1,
               'delegate2': delegate2}
    print('SAVING to', file_name)
    with open(file_name, 'wb') as f:
        dill.dump(to_save, f)
    print('Done saving, now LOADING')
    with open(file_name, 'rb') as f:
        data = dill.load(f)
    print('GOT/loaded', data)
    print('FOR each key')
    o2m = data['one2many']
    del1, del2 = data['delegate1'], data['delegate2']
    print('DEL1, DEL2', del1.delegate, del2.delegate)
    print('COMP1', del1.delegate is o2m)
    print('COMP2', del2.delegate is o2m)
    print('COMP3', del1.delegate is del2.delegate)
    print('KEYS', list(one2many))
    for key in one2many:
        print('KEY', key)
        print(one2many[key], '\nvs\n', o2m[key])
        print('Through delegates')
        print('DELEGATE1')
        print(delegate1[key], 'vs', del1[key])
        print('DELEGATE2')
        print(delegate2[key], 'vs', del2[key])
    # changing o2m should change del1 and/or del2 as well
    o2m['c10'] = [['f11', 'f50'], 'f50']
    print('o2m', o2m)
    print('And for c10 in delegates')
    print('del1', del1['c10'])
    print('del2', del2['c10'])


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
