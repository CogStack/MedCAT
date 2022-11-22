from medcat.utils.data_utils import count_annotations
from medcat.cdb import CDB


def deid_text(cat, text, redact=False):
    new_text = str(text)
    entities = cat.get_entities(text)['entities']
    for ent in sorted(entities.values(), key=lambda ent: ent['start'], reverse=True):
        r = "*"*(ent['end']-ent['start']) if redact else cat.cdb.get_name(ent['cui'])
        new_text = new_text[:ent['start']] + f'[{r}]' + new_text[ent['end']:]
    return new_text


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
