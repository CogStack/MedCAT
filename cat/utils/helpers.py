import numpy as np


def to_json_simple(docs, umls):
    """
    output:  [{'text': <text>, 'entities': [<start,end,type>, ]}]
    """
    d = []

    for doc in docs:
        d.append({'text': doc.text, 'entities': [(e.start_char, e.end_char, umls.tui2name[umls.cui2tui[e.label_]]) for e in doc._.ents]})



def to_json_sumithra(docs, umls):
    """
    output:  [
              [ text, {'entities': [<start,end,type>, ]} ],
              ...]
    """
    d = []

    for doc in docs:
        d.append([doc.text, {'entities': [(e.start_char, e.end_char, umls.tui2name[umls.cui2tui[e.label_]]) for e in doc._.ents]}])

    return d


