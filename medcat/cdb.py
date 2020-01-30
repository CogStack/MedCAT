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
import pandas as pd

log = basic_logger("cdb")
class CDB(object):
    """ Holds all the CDB data required for annotation
    """
    MAX_COO_DICT_SIZE = int(os.getenv('MAX_COO_DICT_SIZE', 10000000))
    MIN_COO_COUNT = int(os.getenv('MIN_COO_COUNT', 100))

    def __init__(self):
        self.index2cui = [] # A list containing all CUIs 
        self.cui2index = {} # Map from cui to index in the index2cui list
        self.name2cui = {} # Converts a normalized concept name to a cui
        self.name2cnt = {} # Converts a normalized concept name to a count
        self.name_isunique = {} # Should this name be skipped
        self.name2original_name = {} # Holds the two versions of a name
        self.name2ntkns = {} # Number of tokens for this name
        self.name_isupper = {} # Checks was this name all upper case in cdb 
        self.cui2desc = {} # Map between a CUI and its cdb description
        self.cui_count = {} # TRAINING - How many times this this CUI appear until now
        self.cui_count_ext = {} # Always - counter for cuis that can be reset, destroyed..
        self.cui2ontos = {} # Cui to ontology from where it comes
        self.cui2names = {} # CUI to all the different names it can have
        self.cui2original_names = {} # CUI to all the different original names it can have
        self.original_name2cuis = {} # Original name to cuis it can be assigned to
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
        self.cui2info = {} # Additional info for a concept
        self.cui_disamb_always = {} # Should this CUI be always disambiguated
        self.vocab = {} # Vocabulary of all words ever, hopefully 
        self._coo_matrix = None # cooccurrence matrix - scikit
        self.coo_dict = {} # cooccurrence dictionary <(cui1, cui2)>:<count>


    def add_concept(self, cui, name, onto, tokens, snames, isupper=False,
                    is_pref_name=False, tui=None, pretty_name='',
                    desc=None, tokens_vocab=None, original_name=None,
                    is_unique=None):
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
        original_name: As in UMLS or any other vocab
        """
        # Add the info property
        if cui not in self.cui2info:
            self.cui2info[cui] = {}

        # Add is name upper
        if name in self.name_isupper:
            self.name_isupper[name] = self.name_isupper[name] or isupper
            self.name_isupper[name] = self.name_isupper[name] or isupper
        else:
            self.name_isupper[name] = isupper

        # Add original name
        if original_name is not None:
            self.name2original_name[name] = original_name

            if original_name in self.original_name2cuis:
                self.original_name2cuis[original_name].add(cui)
            else:
                self.original_name2cuis[original_name] = {cui}

            if cui in self.cui2original_names:
                self.cui2original_names[cui].add(original_name)
            else:
                self.cui2original_names[cui] = {original_name}


        # Add prefered name 
        if is_pref_name:
            self.cui2pref_name[cui] = name
            if pretty_name:
                self.cui2pretty_name[cui] = pretty_name

        if cui not in self.cui2pretty_name and pretty_name:
            self.cui2pretty_name[cui] = pretty_name

        if tui is not None:
            self.cui2tui[cui] = tui

            if tui in self.tui2cuis:
                self.tui2cuis[tui].add(cui)
            else:
                self.tui2cuis[tui] = set([cui])

        if is_unique is not None:
            self.name_isunique[name] = is_unique

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
                self.cui2desc[cui] = str(desc)
            elif str(desc) not in str(self.cui2desc[cui]):
                self.cui2desc[cui] = str(self.cui2desc[cui]) + "\n\n" + str(desc)

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

        # Add number of tokens for this name
        if name in self.name2ntkns:
            self.name2ntkns[name].add(len(tokens))
        else:
            self.name2ntkns[name] = {len(tokens)}

        # Add mappings to onto2cuis
        if onto not in self.onto2cuis:
            self.onto2cuis[onto] = set([cui])
        else:
            self.onto2cuis[onto].add(cui)

        if cui in self.cui2ontos:
            self.cui2ontos[cui].add(onto)
        else:
            self.cui2ontos[cui] = {onto}

        # Add mappings to name2cui
        if name not in self.name2cui:
            self.name2cui[name] = set([cui])
        else:
            self.name2cui[name].add(cui)

        # Add snames to set
        self.sname2name.update(snames)

        # Add mappings to cui2names
        if cui not in self.cui2names:
            self.cui2names[cui] = {name}
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


    def add_tui_names(self, csv_path):
        """ Fils the tui2name dict

        """
        df = pd.read_csv(csv_path)

        for index, row in df.iterrows():
            tui = row['tui']
            name = row['name']
            if tui not in self.tui2name:
                self.tui2name[tui] = name


    def add_context_vec(self, cui, context_vec, negative=False, cntx_type='LONG', inc_cui_count=True, anneal=True, lr=0.5):
        """ Add the vector representation of a context for this CUI

        cui:  The concept in question
        context_vec:  Vector represenation of the context
        negative:  Is this negative context of positive
        cntx_type:  Currently only two supported LONG and SHORT
                     pretty much just based on the window size
        inc_cui_count:  should this be counted
        """
        if cui not in self.cui_count:
            self.increase_cui_count(cui, True)

        # Ignore very similar context
        prob = 0.95


        # Set the right context
        if cntx_type == 'MED':
            cui2context_vec = self.cui2context_vec
        elif cntx_type == 'SHORT':
            cui2context_vec = self.cui2context_vec_short
        elif cntx_type == 'LONG':
            cui2context_vec = self.cui2context_vec_long

        sim = 0
        cv = context_vec
        if cui in cui2context_vec:
            sim = np.dot(unitvec(cv), unitvec(cui2context_vec[cui]))
            if anneal:
                lr = max(lr / self.cui_count[cui], 0.0005)

            if negative:
                b = max(0, sim) * lr
                cui2context_vec[cui] = cui2context_vec[cui]*(1-b) - cv*b
                #cui2context_vec[cui] = cui2context_vec[cui] - cv*b
            else:
                if sim < prob:
                    b = (1 - max(0, sim)) * lr
                    cui2context_vec[cui] = cui2context_vec[cui]*(1-b) + cv*b
                    #cui2context_vec[cui] = cui2context_vec[cui] + cv*b

                    # Increase cui count
                    self.increase_cui_count(cui, inc_cui_count)
        else:
            if negative:
                cui2context_vec[cui] = -1 * cv
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
        # We use done to ignore multiple occ of same concept
        d_cui1 = set()
        pairs = set()
        for i, cui1 in enumerate(cuis):
            if cui1 not in d_cui1:
                for cui2 in cuis[i+1:]:
                    t = cui1+cui2
                    if t not in pairs:
                        self.add_coo(cui1, cui2)
                        pairs.add(t)
                    t = cui2+cui1
                    if t not in pairs:
                        self.add_coo(cui2, cui1)
                        pairs.add(t)
                d_cui1.add(cui1)

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
        self.cui_count_ext = {}
        self.coo_dict = {}
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


    def reset_training(self):
        self.cui_count = {}
        self.cui2context_vec = {}
        self.cui2context_vec_short = {}
        self.cui2context_vec_long = {}
        self.coo_dict = {}
        self.cui_disamb_always = {}
        self.reset_coo_matrix()

    def filter_by_tui(self, tuis_to_keep):
        all_cuis = [c for c_list in [self.tui2cuis[tui] for tui in tuis_to_keep] for c in c_list]
        self.filter_by_cui(all_cuis)


    def filter_by_cui(self, cuis_to_keep=None):
        assert cuis_to_keep, "Cannot remove all concepts, enter at least one CUI in a set."
        print("FYI - with large CDBs this can take a long time.")
        cuis_to_keep = set(cuis_to_keep)
        cuis = []
        print("Gathering CUIs ")
        for cui in self.cui2names:
            if cui not in cuis_to_keep:
                cuis.append(cui)

        print("Cleaning up CUI maps...")
        for i, cui in enumerate(cuis):
            if i % 10000 == 0:
                print(f'removed 10k concepts, {len(cuis) - i} to go...')
            if cui in self.cui2desc:
                del self.cui2desc[cui]
            if cui in self.cui_count:
                del self.cui_count[cui]
            if cui in self.cui_count_ext:
                del self.cui_count_ext[cui]
            if cui in self.cui2names:
                del self.cui2names[cui]
            if cui in self.cui2original_names:
                del self.cui2original_names[cui]
            if cui in self.cui2pref_name:
                del self.cui2pref_name[cui]
            if cui in self.cui2pretty_name:
                del self.cui2pretty_name[cui]
            if cui in self.cui2words:
                del self.cui2words[cui]
            if cui in self.cui2context_vec:
                del self.cui2context_vec[cui]
            if cui in self.cui2context_vec_short:
                del self.cui2context_vec_short[cui]
            if cui in self.cui2context_vec_long:
                del self.cui2context_vec_long[cui]
            if cui in self.cui2info:
                del self.cui2info[cui]
            if cui in self.cui_disamb_always:
                del self.cui_disamb_always[cui]
        print("Done CUI cleaning")

        print("Cleaning names...")
        for name in list(self.name2cui.keys()):
            _cuis = list(self.name2cui[name])

            for cui in _cuis:
                if cui not in cuis_to_keep:
                    self.name2cui[name].remove(cui)

            if len(self.name2cui[name]) == 0:
                del self.name2cui[name]
        print("Done all")


    def print_stats(self):
        """ Print basic statistics on the database
        """
        print("Number of concepts: {:,}".format(len(self.cui2names)))
        print("Number of names:    {:,}".format(len(self.name2cui)))
        print("Number of concepts that received training: {:,}".format(len(self.cui2context_vec)))
        print("Number of seen training examples in total: {:,}".format(sum(self.cui_count.values())))
        print("Average training examples per concept:     {:.1f}".format(np.average(list(self.cui_count.values()))))
