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
        # Put into to_disamb tokens that are not unique
        if name in self.cdb.name_isunique:
            # Is the number of tokens matching for short words
            if not (len(name) < 7 and len(tkns) not in self.cdb.name2ntkns[name]):
                if self.cdb.name_isunique[name]:
                    # Annotate
                    cui = list(self.cdb.name2cui[name])[0]
                    self._cat._add_ann(cui, doc, tkns, acc=1, name=name)
                else:
                    to_disamb.append((list(tkns), name))
            else:
                to_disamb.append((list(tkns), name))
        else:
            # Is the number of tokens matching for short words
            if not (len(name) < 7 and len(tkns) not in self.cdb.name2ntkns[name]):
                if len(self.cdb.name2cui[name]) == 1:
                    # There is only one concept linked to this name
                    cui = list(self.cdb.name2cui[name])[0]
                    self._cat._add_ann(cui, doc, tkns, acc=1, name=name)
                else:
                    to_disamb.append((list(tkns), name))
            else:
                to_disamb.append((list(tkns), name))
