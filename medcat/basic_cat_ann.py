""" I would just ignore this whole class, it's just a lot of rules that work nicely for CDB
once the software is trained the main thing are the context vectors.
"""

import numpy as np
import operator

class CatAnn(object):
    def __init__(self, cdb, spacy_cat):
        self.cdb = cdb
        self._cat = spacy_cat


    def add_ann(self, name, tkns, doc, to_disamb, doc_words):
        # Don't allow concatenation of tokens if len(name) < 5
        if not(len(name) < 5 and len(tkns) > 1):
            if len(self.cdb.name2cui[name]) == 1:
                # Annotate
                cui = list(self.cdb.name2cui[name])[0]
                self._cat._add_ann(cui, doc, tkns, acc=1, name=name)
            else:
                to_disamb.append((list(tkns), name))
        else:
            to_disamb.append((list(tkns), name))
