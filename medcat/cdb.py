""" Representation class for CDB data
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from typing import Dict, List

from medcat.utils.matutils import unitvec, sigmoid
from medcat.utils.attr_dict import AttrDict
from medcat.utils.loggers import basic_logger


class CDB(object):
    """ Concept DataBase - holds all information necessary for NER+L.

    Properties:
        name2cuis (`Dict[str, List[str]]`):
            Map fro concept name to CUIs - one name can map to multiple CUIs.
        name2cuis2status (`Dict[str, Dict[str, str]]`):
            What is the status for a given name and cui pair - each name can be:
                P - Preferred, A - Automatic (e.g. let medcat decide), N - Not common.
        snames (`Set[str]`):
            All possible subnames for all concepts
        cui2names (`Dict[str, Set[str]]`):
            From cui to all names assigned to it. Mainly used for subsetting (maybe even only).
        cui2snames (`Dict[str, Set[str]]`):
            From cui to all sub-names assigned to it. Only used for subsetting.
        cui2context_vector (`Dict[str, Dict[str, np.array]]`):
            From cui to a dictionary of different kinds of context vectors. Normally you would have here
            a short and a long context vector - they are calculated separately.
        cui2count_train (`Dict[str, int]`):
            From CUI to the number of training examples seen.
        cui2tags (`Dict[str, List[str]]`):
            From CUI to a list of tags. This can be used to tag concepts for grouping of whatever.
        cui2type_ids (`Dict[str, Set[str]]`):
            From CUI to type id (e.g. TUI in UMLS).
        cui2preferred_name (`Dict[str, str]`):
            From CUI to the preferred name for this concept.
        addl_info (`Dict[str, Dict[]]`):
            Any additional maps that are not part of the core CDB. These are usually not needed
            for the base NER+L use-case, but can be useufl for Debugging or some special stuff.
        vocab (`Dict[str, int]`):
            Stores all the words tha appear in this CDB and the count for each one.


    """
    MAX_COO_DICT_SIZE = int(os.getenv('MAX_COO_DICT_SIZE', 10000000))
    MIN_COO_COUNT = int(os.getenv('MIN_COO_COUNT', 100))

    def __init__(self):
        self.name2cuis = {}
        self.name2cuis2status = {}

        self.snames = set()

        self.cui2names = {}
        self.cui2snames = {}
        self.cui2context_vector = {}
        self.cui2count_train = {}
        self.cui2tags = {} # Used to add custom tags to CUIs
        self.cui2type_ids = {}
        self.cui2preferred_name = {}

        self.addl_info = {
                'sim_vectors': None,
                'cui2icd10': {},
                'cui2opcs4': {},
                'cui2ontologies': {},
                'cui2original_names': {},
                'cui2description': {},
                'type_id2name': {},
                # Can be extended with whatever is necessary
                }
        self.vocab = {} # Vocabulary of all words ever in our cdb


    def add_name(self, cui: str, names: Dict, name_status: str='A', full_build: bool=False):
        r''' Adds a name to an existing concept.

        Args:
            cui
            names
            name_status
            full_build
        '''
        name_status = name_status.upper()
        if name_status not in ['P', 'A', 'N']:
            # Name status must be one of the three
            name_status = 'A'

        self.add_concept(cui=cui, names=names, ontology='', name_status=name_status, type_ids=set(), description='', full_build=full_build)


    def add_concept(self, cui: str, names: Dict, ontology: str, name_status: str, type_ids: Set[str], description: str, full_build: bool=False):
        r'''
        Add a concept to internal Concept Database (CDB). Depending on what you are providing
        this will add a large number of properties for each concept.

        Args:
            cui (`str`):
                Concept ID or unique identifer in this database, all concepts that have
                the same CUI will be merged internally.
            names (`Dict[str, <object>]`):
                Names for this concept, or the value that if found in free text can be linked to this concept.
                Names is an array like: `[{'tokens': tokens, 'snames': snames, 'raw_name': raw_name}, ...]`
            ontology (`str`):
                Ontology from which the concept is taken (e.g. SNOMEDCT)
            type_ids (`Set[str]`):
                Semantic type identifier (have a look at TUIs in UMLS or SNOMED-CT)
            description (`str`):
                Description of this concept.
            full_build (`bool`, defaults to `False`):
                If True the dictionary self.addl_info will also be populated, contains a lot of extra information
                about concepts, but can be very memory consuming. This is not necessary for normal functioning of MedCAT.
        '''
        # Add CUI to the required dictionaries
        if cui not in self.cui2names:
            # Create placeholders 
            self.cui2names[cui] = set()
            self.cui2snames[cui] = set()
            self.cui2context_vector[cui] = {}
            self.cui2count_train[cui] = 0
            self.cui2tags[cui] = []

            # Add type_ids
            type_ids: self.cui2type_ids[cui] = type_ids

        else:
            # If the CUI is already in update the type_ids
            type_ids: self.cui2type_ids[cui].update(type_ids)

        # Add names to the required dictionaries
        name_info = None
        for name in names:
            name_info = names[name]
            # Extend snames
            self.snames.update(name_info['snames'])

            # Add name to cui2names
            self.cui2names[cui].add(name)
            # Extend cui2snames
            self.cui2snames[cui].update(name_info['snames'])

            if name in self.name2cuis:
                # Means we have alrady seen this name
                if cui not in self.name2cuis[name]:
                    # If CUI is not already linked do it
                    self.name2cuis[name].append(cui)

                    # At the same time it means the cui is also missing from name2cuis2status, but the 
                    #name is there
                    self.name2cuis2status[name][cui] = name_status
            else:
                # Means we never saw this name
                self.name2cuis[name] = [cui]

               # Add name2cuis2status
                self.name2cuis2status[name] = {cui: name_status}


            # Add tokens to vocab
            for token in name_info['tokens']:
                if token in self.vocab:
                    self.vocab[token] += 1
                else:
                    self.vocab[token] = 1

        # Check is this a preferred name for the concept, this takes the name_info
        #dict which must have a value (but still have to check it, just in case).
        if name_info is not None:
            if name_status == 'P':
                self.cui2preferred_name[cui] = name_info['raw_name']
            elif cui not in self.cui2preferred_name:
                # Add the name if it does not exist, this makes the preferred name random
                #for concepts that do not have it.
                self.cui2preferred_name[cui] = name_info['raw_name']

        # Add other fields if full_build
        if full_build:
            # Use ontologies as the base check, anything can be used for this
            if cui not in self.addl_info['cui2ontologies']:
                if ontology: self.addl_info['cui2ontologies'][cui] = set([ontology])
                if description: self.addl_info['cui2description'][cui] = description
                self.addl_info['cui2original_names'][cui] = set([v['raw_name'] for k,v in names.items()])
            else:
                # Update existing ones
                if ontology: self.addl_info['cui2ontologies'][cui].add(ontology)
                if description: self.addl_info['cui2description'][cui] = description
                self.addl_info['cui2original_names'][cui].update([v['raw_name'] for k,v in names.items()])


    def add_type_names(self, type_id2name, reset_existing=False):
        r''' Add type names.

        Args:
            type_id2name (`Dict[str, str]`):
                One or many type_id2name that will be added to this CDB.
        '''
        if reset_existing:
            self.addl_info['type_id2name'] = {}

        self.addl_info['type_id2name'].update(type_id2name)


    def update_context_vector(self, cui, vector, config, cntx_type='LONG', inc_cui_count=True, negative=False):
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


    def import_training(self, cdb, overwrite=True):
        r'''
        This will import vector embeddings from another CDB. No new concept swill be added.
        IMPORTANT it will not import name maps (cui2name or name2cui or ...).

        Args:
            cdb (medcat.cdb.CDB):
                Concept database from which to import training vectors
            overwrite (boolean):
                If True all training data in the existing CDB will be overwritten, else
                the average between the two training vectors will be taken.

        Examples:
            >>> new_cdb.import_traininig(cdb=old_cdb, owerwrite=True)
        '''
        # Import vectors and counts
        for cui in self.cui2names:
            if cui in cdb.cui_count:
                if overwrite or cui not in self.cui_count:
                    self.cui_count[cui] = cdb.cui_count[cui]
                else:
                    self.cui_count[cui] = (self.cui_count[cui] + cdb.cui_count[cui]) / 2

            if cui in cdb.cui2context_vec:
                if overwrite or cui not in self.cui2context_vec:
                    self.cui2context_vec[cui] = cdb.cui2context_vec[cui]
                else:
                    self.cui2context_vec[cui] = (cdb.cui2context_vec[cui] + self.cui2context_vec[cui]) / 2


            if cui in cdb.cui2context_vec_short:
                if overwrite or cui not in self.cui2context_vec_short:
                    self.cui2context_vec_short[cui] = cdb.cui2context_vec_short[cui]
                else:
                    self.cui2context_vec_short[cui] = (cdb.cui2context_vec_short[cui] + self.cui2context_vec_short[cui]) / 2

            if cui in cdb.cui2context_vec_long:
                if overwrite or cui not in self.cui2context_vec_long:
                    self.cui2context_vec_long[cui] = cdb.cui2context_vec_long[cui]
                else:
                    self.cui2context_vec_long[cui] = (cdb.cui2context_vec_long[cui] + self.cui2context_vec_long[cui]) / 2 

            if cui in cdb.cui_disamb_always:
                self.cui_disamb_always[cui] = cdb.cui_disamb_always


    def reset_cui_count(self, n=10):
        r'''
        Reset the CUI count for all concepts that received training, used when starting new unsupervised training
        or for suppervised with annealing.

        Args:
            n (int, optional):
                This will be set as the CUI count for all cuis in this CDB.

        Examples:
            >>> cdb.reset_cui_count()
        '''
        for cui in self.cui_count.keys():
            self.cui_count[cui] = n


    def reset_training(self):
        r'''
        Will remove all training efforts - in other words all embeddings that are learnt
        for concepts in the current CDB. Please note that this does not remove synonyms (names) that were
        potentially added during supervised/online learning.
        '''
        self.cui_count = {}
        self.cui2context_vec = {}
        self.cui2context_vec_short = {}
        self.cui2context_vec_long = {}
        self.coo_dict = {}
        self.cui_disamb_always = {}
        self.reset_coo_matrix()
        self.reset_similarity_matrix()


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


    def reset_similarity_matrix(self):
        self.sim_vectors = None
        self.sim_vectors_counts = None
        self.sim_vectors_tuis = None
        self.sim_vectors_cuis = None


    def most_similar(self, cui, tui_filter=[], min_cnt=0, topn=50):
        r'''
        Given a concept it will calculat what other concepts in this CDB have the most similar
        embedding.

        Args:
            cui (str):
                The concept ID for the base concept for which you want to get the most similar concepts.
            tui_filter (list):
                A list of TUIs that will be used to filterout the returned results. Using this it is possible
                to limit the similarity calculation to only disorders/symptoms/drugs/...
            min_cnt (int):
                Minimum training examples (unsupervised+supervised) that a concept must have to be considered
                for the similarity calculation.
            topn (int):
                How many results to return

        Return:
            results (dict):
                A dictionary with topn results like: {<cui>: {'name': <name>, 'sim': <similarity>, 'tui_name': <tui_name>,
                                                              'tui': <tui>, 'cnt': <number of training examples the concept has seen>}, ...}

        '''
        # Create the matrix if necessary
        if not hasattr(self, 'sim_vectors') or self.sim_vectors is None or len(self.sim_vectors) < len(self.cui2context_vec):
            print("Building similarity matrix")
            log.info("Building similarity matrix")

            sim_vectors = []
            sim_vectors_counts = []
            sim_vectors_tuis = []
            sim_vectors_cuis = []
            for _cui in self.cui2context_vec:
                sim_vectors.append(unitvec(self.cui2context_vec[_cui]))
                sim_vectors_counts.append(self.cui_count[_cui])
                sim_vectors_tuis.append(self.cui2tui.get(_cui, 'unk'))
                sim_vectors_cuis.append(_cui)

            self.sim_vectors = np.array(sim_vectors)
            self.sim_vectors_counts = np.array(sim_vectors_counts)
            self.sim_vectors_tuis = np.array(sim_vectors_tuis)
            self.sim_vectors_cuis = np.array(sim_vectors_cuis)

        # Select appropirate concepts
        tui_inds = np.arange(0, len(self.sim_vectors_tuis))
        if len(tui_filter) > 0:
            tui_inds = np.array([], dtype=np.int32)
            for tui in tui_filter:
                tui_inds = np.union1d(np.where(self.sim_vectors_tuis == tui)[0], tui_inds)
        cnt_inds = np.arange(0, len(self.sim_vectors_counts))
        if min_cnt > 0:
            cnt_inds = np.where(self.sim_vectors_counts >= min_cnt)[0]

        # Intersect cnt and tui
        inds = np.intersect1d(tui_inds, cnt_inds)

        mtrx = self.sim_vectors[inds]
        cuis = self.sim_vectors_cuis[inds]

        sims = np.dot(mtrx, unitvec(self.cui2context_vec[cui]))

        sims_srt = np.argsort(-1*sims)

        # Create the return dict
        res = {}
        for ind, _cui in enumerate(cuis[sims_srt[0:topn]]):
            res[_cui] = {'name': self.cui2pretty_name[_cui], 'sim': sims[sims_srt][ind],
                         'tui_name': self.tui2name.get(self.cui2tui.get(_cui, 'unk'), 'unk'),
                         'tui': self.cui2tui.get(_cui, 'unk'),
                         'cnt': self.cui_count[_cui]}

        return res
