from medcat.utils.data_utils import count_annotations
from medcat.cdb import CDB

from medcat.utils.ner.deid import deid_text as _deid_text
from medcat.utils.decorators import deprecated


@deprecated("API now allows creating a DeId model (medcat.utils.ner.deid.DeIdModel). "
            "It aims to simplify the usage of DeId models. "
            "The use of this model is encouraged over the use of this method.")
def deid_text(*args, **kwargs) -> str:
    return _deid_text(*args, **kwargs)


def make_or_update_cdb(json_path, cdb=None, min_count=0):
    """Creates a new CDB or updates an existing one with new
    concepts if the cdb argument is provided. All concepts that are less frequent
    than min_count will be ignored.
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
