import numpy as np
from spacy.util import escape_html
from .other import *
from medcat.preprocessing.cleaners import clean_name


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


def json2html(doc):
    markup = ""
    offset = 0
    text = doc['text']

    for span in list(doc['entities']):
        start = span['start']
        end = span['end']
        fragments = text[offset:start].split("\n")

        for i, fragment in enumerate(fragments):
            markup += escape_html(fragment)
            if len(fragments) > 1 and i != len(fragments) - 1:
                markup += "</br>"
        ent = {'label': '', 'id': span['id'], 'bg': "rgb(74, 154, 239, {})".format(1 * 1 + 0.12), 'text': escape_html(span['str'])}
        # Add the entity
        markup += TPL_ENT.format(**ent)
        offset = end
    markup += escape_html(text[offset:])

    out = TPL_ENTS.format(content=markup, dir='ltr')

    return out


def prepare_name(cat, name, version='CLEAN'):
    """ Cleans up the name
    """
    name = clean_name(name)

    if version.lower() == 'clean':
        sc_name = cat(name)
        tokens = [str(t.lemma_).lower() for t in sc_name if not t._.is_punct
                  and not t._.to_skip]

    if version.lower() == 'raw':
        sc_name = cat(name)
        tokens = [t.lower_ for t in sc_name if not t._.is_punct
                  and not (t._.to_skip and not t.is_stop)]

    # Join everything and return name 
    name = "".join(tokens)
    return name, tokens


def get_all_from_name(name, nlp, source_value, SEP="", version='clean'):
    sc_name = nlp(source_value)
    name, tokens = prepare_name(nlp, name=name, version=version)
    tokens_vocab = [t.lower_ for t in sc_name if not t._.is_punct]

    snames = []
    sname = ""
    for token in tokens:
        sname = sname + token + SEP
        snames.append(sname.strip())

    return name, tokens, snames, tokens_vocab


def tkn_inds_from_doc(spacy_doc, text_inds=None, source_val=None):
    tkn_inds = None
    start = None
    end = None
    if text_inds is None and source_val in spacy_doc.text:
        start = spacy_doc.text.index(source_val)
        end = start + len(source_val)
    elif text_inds is not None:
        start = text_inds[0]
        end = text_inds[1]

    if start is not None:
        tkn_inds = []
        for tkn in spacy_doc:
            if tkn.idx >= start and tkn.idx <= end:
                tkn_inds.append(tkn.i)

    return tkn_inds
