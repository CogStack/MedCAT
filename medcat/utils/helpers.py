import numpy as np
from spacy.util import escape_html
from .other import *


def to_json_simple(docs, cdb):
    """
    output:  [{'text': <text>, 'entities': [<start,end,type>, ]}]
    """
    d = []

    for doc in docs:
        d.append({'text': doc.text, 'entities': [(e.start_char, e.end_char, cdb.tui2name[cdb.cui2tui[e.label_]]) for e in doc._.ents]})



def to_json_sumithra(docs, cdb):
    """
    output:  [
              [ text, {'entities': [<start,end,type>, ]} ],
              ...]
    """
    d = []

    for doc in docs:
        d.append([doc.text, {'entities': [(e.start_char, e.end_char, cdb.tui2name[cdb.cui2tui[e.label_]]) for e in doc._.ents]}])

    return d


def get_all_from_name(name, nlp, source_value, SEP=""):
    sc_name = nlp(source_value)
    tokens = [str(t.lemma_).lower() for t in sc_name if not t._.is_punct and not t._.to_skip]
    tokens_vocab = [t.lower_ for t in sc_name if not t._.is_punct]

    snames = []
    sname = ""
    for token in tokens:
        sname = sname + token + SEP
        snames.append(sname.strip())

    name = SEP.join(tokens)

    return name, tokens, snames, tokens_vocab


def doc2html(doc):
    markup = ""
    offset = 0
    text = doc.text

    for span in list(doc.ents):
        start = span.start_char
        end = span.end_char
        fragments = text[offset:start].split("\n")

        for i, fragment in enumerate(fragments):
            markup += escape_html(fragment)
            if len(fragments) > 1 and i != len(fragments) - 1:
                markup += "</br>"
        ent = {'label': '', 'id': span._.id, 'bg': "rgb(74, 154, 239, {})".format(span._.acc * span._.acc + 0.12), 'text': escape_html(span.text)}
        # Add the entity
        markup += TPL_ENT.format(**ent)
        offset = end
    markup += escape_html(text[offset:])

    out = TPL_ENTS.format(content=markup, dir='ltr')

    return out
