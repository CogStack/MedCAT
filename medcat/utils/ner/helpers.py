from typing import Callable, Dict, Optional

from medcat.utils.data_utils import count_annotations
from medcat.cdb import CDB


def replace_entities_in_text(text: str,
                             entities: Dict,
                             get_cui_name: Callable[[str], str],
                             redact: bool = False) -> str:
    new_text = str(text)
    for ent in sorted(entities.values(), key=lambda ent: ent['start'], reverse=True):
        r = "*"*(ent['end']-ent['start']
                 ) if redact else get_cui_name(ent['cui'])
        new_text = new_text[:ent['start']] + f'[{r}]' + new_text[ent['end']:]
    return new_text


def make_or_update_cdb(json_path: str, cdb: Optional[CDB] = None,
                       min_count: int = 0) -> CDB:
    """Creates a new CDB or updates an existing one with new
    concepts if the cdb argument is provided. All concepts that are less frequent
    than min_count will be ignored.

    Args:
        json_path (str): The json path
        cdb (Optional[CDB]): The CDB if present. Defaults to None.
        min_count (int): Minimum count to include. Defaults to 0.

    Returns:
        CDB: The same or new CDB.
    """
    cui2cnt = count_annotations(json_path)
    if cdb is None:
        cdb = CDB()

    for cui in cui2cnt.keys():
        if cui2cnt[cui] > min_count:
            # We are adding only what is needed
            cdb.cui2names[cui] = set([cui])
            cdb.cui2preferred_name[cui] = cui

    return cdb
