""" Representation class for UMLS data
"""
import pickle
import numpy as np
from scipy.sparse import dok_matrix
from gensim.matutils import unitvec
from cat.utils.attr_dict import AttrDict
from cat.utils.loggers import basic_logger

log = basic_logger("umls")
MAX_COO_DICT_SIZE = 10000000
MIN_COO_COUNT = 10

class UMLS(object):
    """ Holds all the UMLS data required for annotation

    """
    def __init__(self):
        self.index2cui = []
        self.cui2index = {}
        self.name2cui = {}
        self.name2cnt = {}
        self.name_isupper = {}
        self.cui2desc = {}
        self.cui_count = {}
        self.cui2names = {}
        self.cui2tui = {}
        self.tui2cuis = {}
        self.tui2name = {}
        self.cui2pref_name = {}
        self.cui2pretty_name = {}
        self.sname2name = set()
        self.cui2words = {}
        self.onto2cuis = {}
        self.cui2context_vec = {}
        self.cui2context_vec_short = {}
        self.cui2ncontext_vec = {}
        self.cui2context_words = {}
        self.vocab = {}
        self.cui2scores = {}
        self._coo_matrix = None
        self.coo_dict = {}

        self.CONTEXT_WORDS_LIMIT = 80

    def add_concept(self, cui, name, onto, tokens, snames, isupper, is_pref_name=False, tui=None, pretty_name='',
                    desc=None):
        """ Add a concept to internal UMLS representation

        cui:  Identifier
        name:  Concept name
        onto:  Ontology from which the concept is taken
        tokens:  A list of words existing in the name
        snames:  if name is "heart attack" snames is
                 ['heart', 'heart attack']
        isupper:  If name in the original ontology is upper_cased
        is_pref_name:  If this is the prefered name for this CUI
        tui:  Semantic type
        """
        # Add is name upper
        if name in self.name_isupper:
            self.name_isupper[name] = self.name_isupper[name] or isupper
        else:
            self.name_isupper[name] = isupper

        if is_pref_name:
            self.cui2pref_name[cui] = name
            self.cui2pretty_name[cui] = pretty_name

        if cui not in self.cui2pretty_name:
            self.cui2pretty_name[cui] = pretty_name

        if tui is not None:
            self.cui2tui[cui] = tui

            if tui in self.tui2cuis:
                self.tui2cuis[tui].add(cui)
            else:
                self.tui2cuis[tui] = set([cui])

        # Add name to cnt
        if name not in self.name2cnt:
            self.name2cnt[name] = {}
        if cui in self.name2cnt[name]:
            self.name2cnt[name][cui] += 1
        else:
            self.name2cnt[name][cui] = 1

        # Add description
        if desc is not None:
            if cui not in self.cui2desc:
                self.cui2desc[cui] = desc

        # Add cui to a list of cuis
        if cui not in self.index2cui:
            self.index2cui.append(cui)
            self.cui2index[cui] = len(self.index2cui) - 1

            # Expand coo matrix if it is used
            if self._coo_matrix is not None:
                s = self._coo_matrix.shape[0] + 1
                self._coo_matrix.resize((s, s))

        # Add words to vocab
        for token in tokens:
            if token in self.vocab:
                self.vocab[token] += 1
            else:
                self.vocab[token] = 1

        # Add mappings to onto2cuis
        if onto not in self.onto2cuis:
            self.onto2cuis[onto] = set([cui])
        else:
            self.onto2cuis[onto].add(cui)

        # Add mappings to name2cui
        if name not in self.name2cui:
            self.name2cui[name] = set([cui])
        else:
            self.name2cui[name].add(cui)

        # Add snames to set
        self.sname2name.update(snames)

        # Add mappings to cui2names
        if cui not in self.cui2names:
            self.cui2names[cui] = set([name])
        else:
            self.cui2names[cui].add(name)

        # Add mappings to cui2words
        if cui not in self.cui2words:
            self.cui2words[cui] = {}
        for token in tokens:
            if not token.isdigit() and len(token) > 1:
                if token in self.cui2words[cui]:
                    self.cui2words[cui][token] += 1
                else:
                    self.cui2words[cui][token] = 1


    def add_tui_names(self, d):
        """ Fils the tui2name dict

        d:  map from "tui" to "tui_name"
        """
        for key in d.keys():
            if key not in self.tui2name:
                self.tui2name[key] = d[key]


    def add_context_vec(self, cui, context_vec, negative=False, cntx_type='LONG'):
        """ Add the vector representation of a context for this CUI

        cui:  The concept in question
        context_vec:  Vector represenation of the context
        """

        if cntx_type == 'LONG':
            cui2context_vec = self.cui2context_vec
        elif cntx_type == 'SHORT':
            cui2context_vec = self.cui2context_vec_short

        sim = 0
        cv = context_vec
        if cui in cui2context_vec:
            sim = np.dot(unitvec(cv), unitvec(cui2context_vec[cui]))

            if negative:
                b = max((0.1 / self.cui_count[cui]), 0.000001)  * max(0, sim)
                cui2context_vec[cui] = cui2context_vec[cui]*(1-b) - cv*b
            else:
                if sim < 0.8 and sim > 0.1:
                    c = 0.00001
                    b = max((0.5 / self.cui_count[cui]), c)  * (1 - max(0, sim))
                    cui2context_vec[cui] = cui2context_vec[cui]*(1-b) + cv*b
                elif sim < 0.1:
                    c = 0.0001
                    b = max((0.5 / self.cui_count[cui]), c)  * (1 - max(0, sim))
                    cui2context_vec[cui] = cui2context_vec[cui]*(1-b) + cv*b
        else:
            cui2context_vec[cui] = cv

        return sim


    def add_ncontext_vec(self, cui, ncontext_vec):
        """ Add the vector representation of a context for this CUI

        cui:  The concept in question
        ncontext_vec:  Vector represenation of the context
        """
        if cui in self.cui2ncontext_vec:
            self.cui2ncontext_vec[cui] = (self.cui2ncontext_vec[cui] + ncontext_vec) / 2
        else:
            self.cui2ncontext_vec[cui] = ncontext_vec


    def add_context_words(self, cui, context_words):
        """ Add words that appear in the context of this CUI

        cui:  The concept in question
        context_words:  Array of words that appeard in the context
        """
        if cui in self.cui2context_words:
            vcb = self.cui2context_words[cui]

            for word in context_words:
                if word in vcb:
                    vcb[word] += 1
                else:
                    vcb[word] = 1
        else:
            self.cui2context_words[cui] = {}
            vcb = self.cui2context_words[cui]

            for word in context_words:
                if word in vcb:
                    vcb[word] += 1
                else:
                    vcb[word] = 1

        if len(vcb) > self.CONTEXT_WORDS_LIMIT:
            # Remove 1/3 of the words with lowest frequency
            remove_from = int(self.CONTEXT_WORDS_LIMIT / 3 * 2)
            keys = [k for k in sorted(vcb, key=vcb.get, reverse=True)][remove_from:]
            for key in keys:
                del vcb[key]

            # For the rest reset the counter, most frequent word has len(vcb) / 2 
            #and the rest is in descending order
            pos = 0
            for key, value in sorted(vcb.items(), key=lambda kv: kv[1], reverse=True):
                vcb[key] = max(len(vcb.keys()) - pos, 4) // 2
                pos += 1


    def add_coo(self, cui1, cui2):
        key = (self.cui2index[cui1], self.cui2index[cui2])

        if key in self.coo_dict:
            self.coo_dict[key] += 1
        else:
            self.coo_dict[key] = 1

    def add_coos(self, cuis):
        cnt = 0
        for i, cui1 in enumerate(cuis):
            for cui2 in cuis[i+1:]:
                cnt += 1
                self.add_coo(cui1, cui2)
                self.add_coo(cui2, cui1)

        if len(self.coo_dict) > MAX_COO_DICT_SIZE:
            log.info("Starting the clean of COO_DICT, parameters are\n \
                      MAX_COO_DICT_SIZE: {}\n \
                      MIN_COO_COUNT: {}".format(MAX_COO_DICT_SIZE, MIN_COO_COUNT))

            # Remove entries from coo_dict if too many
            old_size = len(self.coo_dict)
            to_del = []
            for key in self.coo_dict.keys():
                if self.coo_dict[key] < MIN_COO_COUNT:
                    to_del.append(key)

            for key in to_del:
                del self.coo_dict[key]

            new_size = len(self.coo_dict)
            log.info("COO_DICT cleaned, size was: {} and now is {}. In total \
                      {} items were removed".format(old_size, new_size, old_size-new_size))

    @property
    def coo_matrix(self):
        if self._coo_matrix is None:
            s = len(self.cui2index)
            self._coo_matrix = dok_matrix((s, s), dtype=np.uint32)

        self._coo_matrix._update(self.coo_dict)
        return self._coo_matrix


    @coo_matrix.setter
    def coo_matrix(self, val):
        raise AttributeError("Can not set attribute coo_matrix")

    def reset_coo_matrix(self):
        self._coo_matrix = None


    def merge(self, umls):
        """ Merges another umls instance into this one

        umls:  To be merged with this one
        """

        # Just an extension
        self.index2cui.extend(umls.index2cui)

        # cui2index has to be rebuilt
        self.cui2index = {}
        for ind, word in enumerate(self.index2cui):
            self.cui2index[word] = ind

        # name2cui - new names should be added and old extended
        for key in umls.name2cui.keys():
            if key in self.name2cui:
                self.name2cui[key].update(umls.name2cui[key])
            else:
                self.name2cui[key] = umls.name2cui[key]

        # name2cnt - old rewriten by new
        self.name2cnt.update(umls.name2cnt)
        # name_isupper - update dict
        self.name_isupper.update(umls.name_isupper)

        # cui2names - new names should be added and old extended
        for key in umls.cui2names.keys():
            if key in self.cui2names:
                self.cui2names[key].update(umls.cui2names[key])
            else:
                self.cui2names[key] = umls.cui2names[key]

        # Just update the dictionaries
        self.cui2tui.update(umls.cui2tui)
        self.cui2pref_name.update(umls.cui2pref_name)
        # Update the set
        self.sname2name.update(umls.sname2name)

        # cui2words - new words should be added
        for key in umls.cui2words.keys():
            if key in self.cui2words:
                self.cui2words[key].update(umls.cui2words[key])
            else:
                self.cui2words[key] = umls.cui2words[key]

        # onto2cuis - new words should be added
        for key in umls.onto2cuis.keys():
            if key in self.onto2cuis:
                self.onto2cuis[key].update(umls.onto2cuis[key])
            else:
                self.onto2cuis[key] = umls.onto2cuis[key]

        # Just update vocab
        self.vocab.update(umls.vocab)
        # Remove coo_matrix if it exists
        self._coo_matrix = None
        # Merge tui2cuis, if they exist
        if hasattr(umls, 'tui2cuis'):
            self.tui2cuis.update(umls.tui2cuis)

        # Merge the training part
        self.merge_train(umls)


    def get_train_dict(self):
        return {'cui2context_vec': self.cui2context_vec,
                'cui2context_words': self.cui2context_words,
                'cui_count': self.cui_count,
                'coo_dict': self.coo_dict,
                'cui2ncontext_vec': self.cui2ncontext_vec}


    def merge_train_dict(self, t_dict):
        attr_dict = AttrDict()
        attr_dict.update(t_dict)
        self.merge_train(attr_dict)


    def merge_train(self, umls):
        # To be merged: cui2context_vec, cui2context_words, cui_count, coo_dict
        #cui2ncontext_vec

        # Merge cui2context_vec
        for key in umls.cui2context_vec.keys():
            if key in self.cui2context_vec:
                self.cui2context_vec[key] = (self.cui2context_vec[key] + umls.cui2context_vec[key]) / 2
            else:
                self.cui2context_vec[key] = umls.cui2context_vec[key]

        # Merge cui2context_vec
        for key in umls.cui2ncontext_vec.keys():
            if key in self.cui2ncontext_vec:
                self.cui2ncontext_vec[key] = (self.cui2ncontext_vec[key] + umls.cui2ncontext_vec[key]) / 2
            else:
                self.cui2ncontext_vec[key] = umls.cui2ncontext_vec[key]

        # Merge cui2context_words
        for cui in umls.cui2context_words.keys():
            if cui in self.cui2context_words:
                for word in umls.cui2context_words[cui]:
                    if word in self.cui2context_words[cui]:
                        self.cui2context_words[cui][word] += umls.cui2context_words[cui][word]
                    else:
                        self.cui2context_words[cui][word] = umls.cui2context_words[cui][word]
            else:
                self.cui2context_words[cui] = umls.cui2context_words[cui]

        # Merge coo_dict
        for key in umls.coo_dict.keys():
            if key in self.coo_dict:
                self.coo_dict[key] += umls.coo_dict[key]
            else:
                self.coo_dict[key] = umls.coo_dict[key]


        # Merge cui_count
        for key in umls.cui_count.keys():
            if key in self.cui_count:
                self.cui_count[key] += umls.cui_count[key]
            else:
                self.cui_count[key] = umls.cui_count[key]


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


    def save_dict(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)


    def load_dict(self, path):
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)
