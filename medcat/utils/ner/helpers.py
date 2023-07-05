from medcat.utils.data_utils import count_annotations
from medcat.cdb import CDB

from medcat.utils.decorators import deprecated


# For now, we will keep this method separate from the above class
# This is so that we wouldn't need to create a thorwaway object
# when calling the method from .helpers where it used to be.
# After the deprecated method in .helpers is removed, we can
# move this to a proper class method.
def _deid_text(cat, text: str, redact: bool = False) -> str:
    """De-identify text.

    De-identified text.
    If redaction is enabled, identifiable entities will be
    replaced with starts (e.g `*****`).
    Otherwise, the replacement will be the CUI or in other words,
    the type of information that was hidden (e.g [PATIENT]).


    Args:
        cat (CAT): The CAT object to use for deid.
        text (str): The input document.
        redact (bool, optional): Whether to redact. Defaults to False.

    Returns:
        str: The de-identified document.
    """
    new_text = str(text)
    entities = cat.get_entities(text)['entities']
    for ent in sorted(entities.values(), key=lambda ent: ent['start'], reverse=True):
        r = "*"*(ent['end']-ent['start']
                 ) if redact else cat.cdb.get_name(ent['cui'])
        new_text = new_text[:ent['start']] + f'[{r}]' + new_text[ent['end']:]
    return new_text


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
