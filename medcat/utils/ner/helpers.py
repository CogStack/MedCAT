from typing import Callable, Dict, List
from medcat.utils.data_utils import count_annotations
from medcat.cdb import CDB
from medcat.utils.decorators import deprecated
from medcat.utils.ner.chunking import get_chunks
from spacy.pipeline.ner import EntityRecognizer


# For now, we will keep this method separate from the above class
# This is so that we wouldn't need to create a thorwaway object
# when calling the method from .helpers where it used to be.
# After the deprecated method in .helpers is removed, we can
# move this to a proper class method.
def _deid_text(cat, text: str, redact: bool = False, chunk: bool = True) -> str:
    """De-identify text.

    De-identified text.
    If redaction is enabled, identifiable entities will be
    replaced with starts (e.g `*****`).
    Otherwise, the replacement will be the CUI or in other words,
    the type of information that was hidden (e.g [PATIENT]).
    v2: Updated to add support for chunking


    Args:
        cat (CAT): The CAT object to use for deid.
        text (str): The input document.
        redact (bool, optional): Whether to redact. Defaults to False.

    Returns:
        str: The de-identified document.
    """
    if chunk:
        de_id_pipe: EntityRecognizer
        de_id_pipe = cat.pipe._nlp.get_pipe("deid")
        chunked_text = get_chunks(text, de_id_pipe.ner_pipe.tokenizer)
        anon_text: List[str] = []
        for text_ in chunked_text:
            entities = cat.get_entities(text_)['entities']
            anon_ = replace_entities_in_text(text_, entities, cat.cdb.get_name, redact=redact)
            anon_text.append(anon_)
        return "".join(anon_text)
    else:
        entities = cat.get_entities(text)['entities']
        return replace_entities_in_text(text, entities, cat.cdb.get_name, redact=redact)


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
