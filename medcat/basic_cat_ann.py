""" I would just ignore this whole class, it's just a lot of rules that work nicely for CDB
once the software is trained the main thing are the context vectors.
"""
import numpy as np
import operator

class CatAnn(object):
    def __init__(self, cdb, spacy_cat):
        self.cdb = cdb
        self._cat = spacy_cat
        self.pref_names = set(cdb.cui2pref_name.values())


    def add_ann(self, name, tkns, doc, to_disamb, doc_words):
        one_tkn_upper = False
        if len(tkns) == 1 and tkns[0].is_upper:
            one_tkn_upper = True

        if len(name) > 1 or one_tkn_upper:
            if name in self.cdb.name_isunique:
                # Is the number of tokens matching for short words
                if len(name) >= 7 or len(tkns) in self.cdb.name2ntkns[name]:
                    if self.cdb.name_isunique[name]:
                        # Annotate
                        cui = list(self.cdb.name2cui[name])[0]
                        self._cat._add_ann(cui, doc, tkns, acc=1, name=name)
                    else:
                        to_disamb.append((list(tkns), name))
                else:
                    # For now ignore if < 7 and tokens don't match
                    #to_disamb.append((list(tkns), name))
                    pass
            else:
                # Is the number of tokens matching for short words
                if len(name) > 7 or len(tkns) in self.cdb.name2ntkns[name]:
                    if len(self.cdb.name2cui[name]) == 1 and len(name) > 2:
                        # There is only one concept linked to this name and has
                        #more than 2 characters
                        cui = list(self.cdb.name2cui[name])[0]
                        self._cat._add_ann(cui, doc, tkns, acc=1, name=name)
                    elif self._cat.train and name in self.pref_names and len(name) > 3:
                        # If training use prefered names as ground truth
                        cuis = self.cdb.name2cui[name]
                        for cui in cuis:
                            if name == self.cdb.cui2pref_name.get(cui, 'nan-nan'):
                                self._cat._add_ann(cui, doc, tkns, acc=1, name=name)
                    else:
                        to_disamb.append((list(tkns), name))
                else:
                    # For now ignore
                    #to_disamb.append((list(tkns), name))
                    pass
