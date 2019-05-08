""" Representation class for CDB data
"""
import pickle
import numpy as np
from scipy.sparse import dok_matrix
#from gensim.matutils import unitvec
from medcat.utils.matutils import unitvec, sigmoid
from medcat.utils.attr_dict import AttrDict
from medcat.utils.loggers import basic_logger
import os

log = basic_logger("cdb")
class CDB(object):
    """ Holds all the CDB data required for annotation
    """
    MAX_COO_DICT_SIZE = int(os.getenv('MAX_COO_DICT_SIZE', 10000000))
    MIN_COO_COUNT = int(os.getenv('MIN_COO_COUNT', 100))

    def __init__(self):
        self.index2cui = [] # A list containing all CUIs 
        self.cui2index = {} # Map from cui to index in the index2cui list
        self.is_unique = {} # Is a name unique
        self.name2cui = {} # Converts a normalized concept name to a cui
        self.name2cnt = {} # Converts a normalized concept name to a count
        self.name_isupper = {} # Checks was this name all upper case in cdb 
        self.cui2desc = {} # Map between a CUI and its cdb description
        self.cui_count = {} # TRAINING - How many times this this CUI appear until now
        self.cui_count_ext = {} # Always - counter for cuis that can be reset, destroyed..
        self.cui2names = {} # CUI to all the different names it can have
        self.cui2tui = {} # CUI to the semantic type ID
        self.tui2cuis = {} # Semantic type id to a list of CUIs that have it
        self.tui2name = {} # Semnatic tpye id to its name
        self.cui2pref_name = {} # Get the prefered name for a CUI - taken from CDB 
        self.cui2pretty_name = {} # Get the pretty name for a CUI - taken from CDB 
        self.sname2name = set() # Internal - subnames to nam
        self.cui2words = {} # CUI to all the words that can describe it
        self.onto2cuis = {} # Ontology to all the CUIs contained in it
        self.cui2context_vec = {} # CUI to context vector
        self.cui2context_vec_short = {} # CUI to context vector - short
        self.cui2context_vec_long = {} # CUI to context vector - long
        self.cui2ncontext_vec = {} # CUI to negative context vector
        self.vocab = {} # Vocabulary of all words ever, hopefully 
        self._coo_matrix = None # cooccurrence matrix - scikit
        self.coo_dict = {} # cooccurrence dictionary <(cui1, cui2)>:<count>


    def add_concept(self, cui, name, onto, tokens, snames, isupper=False, is_pref_name=False, tui=None, pretty_name='',
                    desc=None, tokens_vocab=None, unique=True):
        """ Add a concept to internal CDB representation

        cui:  Identifier
        name:  Concept name
        onto:  Ontology from which the concept is taken
        tokens:  A list of words existing in the name
        snames:  if name is "heart attack" snames is
                 ['heart', 'heart attack']
        isupper:  If name in the original ontology is upper_cased
        is_pref_name:  If this is the prefered name for this CUI
        tui:  Semantic type
        pretty_name:  Pretty name for this concept
        desc:  Description of this concept - can take a lot of space
        tokens_vocab:  Tokens that should be added to the vocabulary, usually not normalized
        """
        # Add is name upper
        if name in self.name_isupper:
            self.name_isupper[name] = self.name_isupper[name] or isupper
        else:
            self.name_isupper[name] = isupper

        # Add prefered name 
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
        for token in tokens_vocab:
            if token in self.vocab:
                self.vocab[token] += 1
            else:
                self.vocab[token] = 1
        # Add also the normalized tokens, why not
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


    def add_context_vec(self, cui, context_vec, negative=False, cntx_type='LONG', inc_cui_count=True):
        """ Add the vector representation of a context for this CUI

        cui:  The concept in question
        context_vec:  Vector represenation of the context
        negative:  Is this negative context of positive
        cntx_type:  Currently only two supported LONG and SHORT
                     pretty much just based on the window size
        inc_cui_count:  should this be counted
        """
        prob = 0.5
        if cntx_type == 'MED':
            cui2context_vec = self.cui2context_vec
        elif cntx_type == 'SHORT':
            cui2context_vec = self.cui2context_vec_short
        elif cntx_type == 'LONG':
            cui2context_vec = self.cui2context_vec_long
            prob = 2


        sim = 0
        cv = context_vec
        if cui in cui2context_vec:
            # Just in case cui_count was not set
            if cui not in self.cui_count:
                self.increase_cui_count(cui, True)

            sim = np.dot(unitvec(cv), unitvec(cui2context_vec[cui]))
            #sim = sigmoid(np.dot(cv, cui2context_vec[cui]))

            if negative:
                b = max((0.2 / self.cui_count[cui]), 0.0001)  * max(0, sim)
                cui2context_vec[cui] = cui2context_vec[cui]*(1-b) - cv*b
                #cui2context_vec[cui] = cui2context_vec[cui] - cv*b

            else:
                if sim < prob:
                    c = 0.001
                    b = max((0.3 / self.cui_count[cui]), c)  * (1 - max(0, sim))
                    cui2context_vec[cui] = cui2context_vec[cui]*(1-b) + cv*b
                    #cui2context_vec[cui] = cui2context_vec[cui] + cv*b

                    # Increase cui count
                    self.increase_cui_count(cui, inc_cui_count)
        else:
            cui2context_vec[cui] = cv
            self.increase_cui_count(cui, inc_cui_count)

        return sim


    def increase_cui_count(self, cui, inc_cui_count):
        if inc_cui_count:
            if cui in self.cui_count:
                self.cui_count[cui] += 1
            else:
                self.cui_count[cui] = 1


    def add_ncontext_vec(self, cui, ncontext_vec):
        """ Add the vector representation of a context for this CUI

        cui:  The concept in question
        ncontext_vec:  Vector represenation of the context
        """

        sim = 0
        cv = ncontext_vec
        cui2context_vec = self.cui2ncontext_vec

        if cui in self.cui_count:
            if cui in cui2context_vec:
                sim = np.dot(unitvec(cv), unitvec(cui2context_vec[cui]))
                c = 0.001
                b = max((0.5 / self.cui_count[cui]), c)  * (1 - max(0, sim))
                cui2context_vec[cui] = cui2context_vec[cui]*(1-b) + cv*b
            else:
                cui2context_vec[cui] = cv


    def add_coo(self, cui1, cui2):
        """ Add one cooccurrence

        cui1:  Base CUI
        cui2:  Coocured with CUI
        """
        key = (self.cui2index[cui1], self.cui2index[cui2])

        if key in self.coo_dict:
            self.coo_dict[key] += 1
        else:
            self.coo_dict[key] = 1


    def add_coos(self, cuis):
        """ Given a list of CUIs it will add them to the coo matrix
        saying that each CUI cooccurred with each one

        cuis:  List of CUIs
        """
        cnt = 0
        for i, cui1 in enumerate(cuis):
            for cui2 in cuis[i+1:]:
                cnt += 1
                self.add_coo(cui1, cui2)
                self.add_coo(cui2, cui1)

        if len(self.coo_dict) > self.MAX_COO_DICT_SIZE:
            log.info("Starting the clean of COO_DICT, parameters are\n \
                      MAX_COO_DICT_SIZE: {}\n \
                      MIN_COO_COUNT: {}".format(self.MAX_COO_DICT_SIZE, self.MIN_COO_COUNT))

            # Remove entries from coo_dict if too many
            old_size = len(self.coo_dict)
            to_del = []
            for key in self.coo_dict.keys():
                if self.coo_dict[key] < self.MIN_COO_COUNT:
                    to_del.append(key)

            for key in to_del:
                del self.coo_dict[key]

            new_size = len(self.coo_dict)
            log.info("COO_DICT cleaned, size was: {} and now is {}. In total \
                      {} items were removed".format(old_size, new_size, old_size-new_size))

    @property
    def coo_matrix(self):
        """ Get the COO Matrix as scikit dok_matrix
        """
        if self._coo_matrix is None:
            s = len(self.cui2index)
            self._coo_matrix = dok_matrix((s, s), dtype=np.uint32)

        self._coo_matrix._update(self.coo_dict)
        return self._coo_matrix


    @coo_matrix.setter
    def coo_matrix(self, val):
        """ Imposible to set, it is built internally
        """
        raise AttributeError("Can not set attribute coo_matrix")

    def reset_coo_matrix(self):
        """ Remove the COO-Matrix
        """
        self._coo_matrix = None


    def merge_run_only(self, coo_dict, cui_count_ext):
        """ Merges only the coo matrix and cui_count_ext
        """
        # Reset the coo matrix as it is not valid anymore
        self.reset_coo_matrix()

        # Merge coo_dict
        for key in coo_dict.keys():
            if key in self.coo_dict:
                self.coo_dict[key] += coo_dict[key]
            else:
                self.coo_dict[key] = coo_dict[key]


        # Merge cui_count_ext
        for key in cui_count_ext.keys():
            if key in self.cui_count_ext:
                self.cui_count_ext[key] += cui_count_ext[key]
            else:
                self.cui_count_ext[key] = cui_count_ext[key]


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


    def save_dict(self, path):
        """ Saves variables of this object
        """
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)


    def load_dict(self, path):
        """ Loads variables of this object
        """
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)


    def filter(self, tuis=[], cuis=[]):
        """ A fairly simple function that is limiting the CDB to only certain cuis or tuis

        tuis:  List of tuis to filter by
        cuis:  List of cuis to filter by
        """
        # TODO: This one has to be fixed, but not so easy, for now this is good enough
        # self.sname2name = set()

        # Just reset the co-matrix and the coo_dict
        self._coo_matrix = None
        self.coo_dict = {}

        _cuis = set(cuis)
        if len(tuis) > 0:
            for tui in tuis:
                _cuis.update(self.tui2cuis[tui])

        # Remove everything but the cuis in _cuis from everywhere 
        tmp = self.index2cui
        self.index2cui = []
        self.cui2index = {}
        # cui_count
        tmp_cui_count = self.cui_count
        self.cui_count = {}
        # cui2desc
        tmp_cui2desc = self.cui2desc
        self.cui2desc = {}
        # cui2tui
        tmp_cui2tui = self.cui2tui
        self.cui2tui = {}
        # cui2pref_name
        tmp_cui2pref_name = self.cui2pref_name
        self.cui2pref_name = {}
        # cui2pretty_name = {}
        tmp_cui2pretty_name = self.cui2pretty_name
        self.cui2pretty_name = {}
        # cui2words
        tmp_cui2words = self.cui2words
        self.cui2words = {}
        # cui2context_vec 
        tmp_cui2context_vec = self.cui2context_vec
        self.cui2context_vec = {}
        # cui2context_vec_short
        tmp_cui2context_vec_short = self.cui2context_vec_short
        self.cui2context_vec_short = {}
        # cui2ncontext_vec
        tmp_cui2ncontext_vec = self.cui2ncontext_vec
        self.cui2ncontext_vec = {}
        tmp_cui2names = self.cui2names
        self.cui2names = {}
        for cui in tmp:
            if cui in _cuis:
                self.index2cui.append(cui)
                self.cui2index[cui] = len(self.index2cui) - 1
                if cui in tmp_cui2desc:
                    self.cui2desc[cui] = tmp_cui2desc[cui]
                if cui in tmp_cui_count:
                    self.cui_count[cui] = tmp_cui_count[cui]
                if cui in tmp_cui2tui:
                    self.cui2tui[cui] = tmp_cui2tui[cui]
                if cui in tmp_cui2pref_name:
                    self.cui2pref_name[cui] = tmp_cui2pref_name[cui]
                self.cui2pretty_name[cui] = tmp_cui2pretty_name[cui]
                self.cui2words[cui] = tmp_cui2words[cui]
                if cui in tmp_cui2context_vec:
                    self.cui2context_vec[cui] = tmp_cui2context_vec[cui]
                if cui in tmp_cui2context_vec_short:
                    self.cui2context_vec_short[cui] = tmp_cui2context_vec_short[cui]
                if cui in tmp_cui2ncontext_vec:
                    self.cui2ncontext_vec[cui] = tmp_cui2ncontext_vec[cui]
                self.cui2names[cui] = tmp_cui2names[cui]

        # The main one name2cui
        tmp = self.name2cui
        self.name2cui = {}
        # name2cnt
        tmp_name2cnt = self.name2cnt
        self.name2cnt = {}
        # name_isupper
        tmp_name_isupper = self.name_isupper
        self.name_isupper = {}
        for name, cuis in tmp.items():
            for cui in cuis:
                if cui in _cuis:
                    if cui in self.name2cui:
                        self.name2cui[name].add(cui)
                    else:
                        self.name2cui[name] = set([cui])
                    # name2cnt
                    self.name2cnt[name] = tmp_name2cnt[name]
                    # name_isupper
                    self.name_isupper[name] = tmp_name_isupper[name]


    def reset_training(self):
        self.cui_count = {}
        self.cui2context_vec = {}
        self.cui2ncontext_vec = {}
        self.cui2context_vec_short = {}
        self.cui2context_vec_long = {}
        self.coo_dict = {}
        self.cui2ncontext_vec = {}
        self.reset_coo_matrix()
