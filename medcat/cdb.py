""" Representation class for CDB data
"""

import dill
import logging
import numpy as np
from typing import Dict, List, Set

from medcat.utils.matutils import unitvec, sigmoid
from medcat.utils.ml_utils import get_lr_linking
from medcat.config import Config


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
        cui2context_vectors (`Dict[str, Dict[str, np.array]]`):
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
        cui2average_confidence(`Dict[str, str]`):
            Used for dynamic thresholding. Holds the average confidence for this CUI given the training examples.
        name2count_train(`Dict[str, str]`):
            Counts how often did a name appear during training.
        addl_info (`Dict[str, Dict[]]`):
            Any additional maps that are not part of the core CDB. These are usually not needed
            for the base NER+L use-case, but can be useufl for Debugging or some special stuff.
        vocab (`Dict[str, int]`):
            Stores all the words tha appear in this CDB and the count for each one.
    """
    log = logging.getLogger(__name__)
    def __init__(self, config):
        self.config = config
        self.name2cuis = {}
        self.name2cuis2status = {}

        self.snames = set()

        self.cui2names = {}
        self.cui2snames = {}
        self.cui2context_vectors = {}
        self.cui2count_train = {}
        self.cui2tags = {} # Used to add custom tags to CUIs
        self.cui2type_ids = {}
        self.cui2preferred_name = {}
        self.cui2average_confidence = {}
        self.name2count_train = {}

        self.addl_info = {
                'cui2icd10': {},
                'cui2opcs4': {},
                'cui2ontologies': {},
                'cui2original_names': {},
                'cui2description': {},
                'type_id2name': {},
                'type_id2cuis': {},
                'cui2group': {},
                # Can be extended with whatever is necessary
                }
        self.vocab = {} # Vocabulary of all words ever in our cdb
        self._optim_params = None


    def update_cui2_average_confidence(self, cui, new_sim):
        self.cui2average_confidence[cui] = (self.cui2average_confidence[cui] * self.cui2count_train[cui] + new_sim)  / (self.cui2count_train[cui] + 1)

    def remove_names(self, cui: str, names: Dict):
        r''' Remove names from an existing concept - efect is this name will never again be used to link to this concept.
        This will only remove the name from the linker (namely name2cuis and name2cuis2status), the name will still be present everywhere else.
        Why? Because it is bothersome to remove it from everywhere, but
        could also be useful to keep the removed names in e.g. cui2names.

        Args:
            cui (`str`):
                Concept ID or unique identifer in this database.
            names (`Dict[str, Dict]`):
                Names to be removed, should look like: `{'name': {'tokens': tokens, 'snames': snames, 'raw_name': raw_name}, ...}`
        '''
        for name in names.keys():
            if name in self.name2cuis:
                if cui in self.name2cuis[name]:
                    self.name2cuis[name].remove(cui)
                if len(self.name2cuis[name]) == 0:
                    del self.name2cuis[name]

            # Remove from name2cuis2status
            if name in self.name2cuis2status:
                if cui in self.name2cuis2status[name]:
                    _ = self.name2cuis2status[name].pop(cui)
                if len(self.name2cuis2status[name]) == 0:
                    del self.name2cuis2status[name]

            # Set to disamb always if name2cuis2status is now only one CUI
            if name in self.name2cuis2status:
                if len(self.name2cuis2status[name]) == 1:
                    for _cui in self.name2cuis2status[name]:
                        if self.name2cuis2status[name][_cui] == 'A':
                            self.name2cuis2status[name][_cui] = 'N'
                        elif self.name2cuis2status[name][_cui] == 'P':
                            self.name2cuis2status[name][_cui] = 'PD'


    def add_names(self, cui: str, names: Dict, name_status: str='A', full_build: bool=False):
        r''' Adds a name to an existing concept.

        Args:
            cui (`str`):
                Concept ID or unique identifer in this database, all concepts that have
                the same CUI will be merged internally.
            names (`Dict[str, Dict]`):
                Names for this concept, or the value that if found in free text can be linked to this concept.
                Names is an dict like: `{name: {'tokens': tokens, 'snames': snames, 'raw_name': raw_name}, ...}`
            name_status (`str`):
                One of `P`, `N`, `A`
            full_build (`bool`, defaults to `False`):
                If True the dictionary self.addl_info will also be populated, contains a lot of extra information
                about concepts, but can be very memory consuming. This is not necessary for normal functioning of MedCAT.
        '''
        name_status = name_status.upper()
        if name_status not in ['P', 'A', 'N']:
            # Name status must be one of the three
            name_status = 'A'

        self.add_concept(cui=cui, names=names, ontologies=set(), name_status=name_status, type_ids=set(), description='', full_build=full_build)


    def add_concept(self, cui: str, names: Dict, ontologies: set(), name_status: str, type_ids: Set[str], description: str, full_build: bool=False):
        r'''
        Add a concept to internal Concept Database (CDB). Depending on what you are providing
        this will add a large number of properties for each concept.

        Args:
            cui (`str`):
                Concept ID or unique identifer in this database, all concepts that have
                the same CUI will be merged internally.
            names (`Dict[str, Dict]`):
                Names for this concept, or the value that if found in free text can be linked to this concept.
                Names is an dict like: `{name: {'tokens': tokens, 'snames': snames, 'raw_name': raw_name}, ...}`
            ontologies(`Set[str]`):
                ontologies in which the concept exists (e.g. SNOMEDCT, HPO)
            name_status (`str`):
                One of `P`, `N`, `A`
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
            self.cui2context_vectors[cui] = {}
            self.cui2count_train[cui] = 0
            self.cui2average_confidence[cui] = 0
            self.cui2tags[cui] = []

            # Add type_ids
            self.cui2type_ids[cui] = type_ids
        else:
            # If the CUI is already in update the type_ids
            self.cui2type_ids[cui].update(type_ids)

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
            # Add name to counts
            self.name2count_train[name] = 0

            if name in self.name2cuis:
                # Means we have alrady seen this name
                if cui not in self.name2cuis[name]:
                    # If CUI is not already linked do it
                    self.name2cuis[name].append(cui)

                    # At the same time it means the cui is also missing from name2cuis2status, but the 
                    #name is there
                    self.name2cuis2status[name][cui] = name_status
                elif name_status == 'P':
                    # If name_status is P overwrite whatever was the old status
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
            if name_status == 'P' and cui not in self.cui2preferred_name:
                # Do not overwrite old preferred names
                self.cui2preferred_name[cui] = name_info['raw_name']

        # Add other fields if full_build
        if full_build:
            # Use original_names as the base check because they must be added
            if cui not in self.addl_info['cui2original_names']:
                if ontologies: self.addl_info['cui2ontologies'][cui] = ontologies
                if description: self.addl_info['cui2description'][cui] = description
                self.addl_info['cui2original_names'][cui] = set([v['raw_name'] for k,v in names.items()])
            else:
                # Update existing ones
                if ontologies: self.addl_info['cui2ontologies'][cui].update(ontologies)
                if description: self.addl_info['cui2description'][cui] = description
                self.addl_info['cui2original_names'][cui].update([v['raw_name'] for k,v in names.items()])

            for type_id in type_ids:
                # Add type_id2cuis link
                if type_id in self.addl_info['type_id2cuis']:
                    self.addl_info['type_id2cuis'][type_id].add(cui)
                else:
                    self.addl_info['type_id2cuis'][type_id] = {cui}


    def add_addl_info(self, name, data, reset_existing=False):
        r''' Add data to the addl_info dictionary. This is done in a function to
        not directly access the addl_info dictionary.

        Args:
            name (`str`):
                What key should be used in the `addl_info` dictionary.
            data (`Dict[<whatever>]`):
                What will be added as the value for the key `name`
            reset_existing (`bool`):
                Should old data be removed if it exists
        '''
        if reset_existing:
            self.addl_info[name] = {}

        self.addl_info[name].update(data)


    def update_context_vector(self, cui, vectors, negative=False, lr=None, cui_count=0):
        r''' Add the vector representation of a context for this CUI.

        cui (`str`):
            The concept in question.
        vectors (`Dict[str, np.array]`):
            Vector represenation of the context, must have the format: {'context_type': np.array(<vector>), ...}
            context_type - is usually one of: ['long', 'medium', 'short']
        negative (`bool`, defaults to `False`):
            Is this negative context of positive.
        lr (`int`, optional):
            If set it will override the base value from the config file.
        cui_count (`int`, defaults to 0):
            The learning rate will be calculated based on the count for the provided CUI + cui_count.
        '''
        similarity = None
        if cui in self.cui2context_vectors:
            for context_type, vector in vectors.items():
                # Get the right context
                if context_type in self.cui2context_vectors[cui]:
                    cv = self.cui2context_vectors[cui][context_type]
                    similarity = np.dot(unitvec(cv), unitvec(vector))

                    # Get the learning rate if None
                    if lr is None:
                        lr = get_lr_linking(self.config, self.cui2count_train[cui] + cui_count, self._optim_params, similarity)

                    if negative:
                        # Add negative context
                        b = max(0, similarity) * lr
                        self.cui2context_vectors[cui][context_type] = cv*(1-b) - vector*b
                    else:
                        b = (1 - max(0, similarity)) * lr
                        self.cui2context_vectors[cui][context_type] = cv*(1-b) + vector*b

                    # DEBUG
                    self.log.debug("Updated vector embedding.\n" + \
                            "CUI: {}, Context Type: {}, Similarity: {:.2f}, Is Negative: {}, LR: {:.5f}, b: {:.3f}".format(cui, context_type,
                                similarity, negative, lr, b))
                    cv = self.cui2context_vectors[cui][context_type]
                    similarity_after = np.dot(unitvec(cv), unitvec(vector))
                    self.log.debug("Similarity before vs after: {:.5f} vs {:.5f}".format(similarity, similarity_after))
                else:
                    if negative:
                        self.cui2context_vectors[cui][context_type] = -1 * vector
                    else:
                        self.cui2context_vectors[cui][context_type] = vector

                    # DEBUG
                    self.log.debug("Added new context type with vectors.\n" + \
                            "CUI: {}, Context Type: {}, Is Negative: {}".format(cui, context_type, negative))

        if not negative:
            # Increase counter only for positive examples
            self.cui2count_train[cui] += 1


    def save(self, path):
        r''' Saves model to file (in fact it saves vairables of this class). 

        Args:
            path (`str`):
                Path to a file where the model will be saved
        '''
        with open(path, 'wb') as f:
            # No idea how to this correctly
            to_save = {}
            to_save['config'] = self.config.__dict__
            to_save['cdb'] = {k:v for k,v in self.__dict__.items() if k != 'config'}
            dill.dump(to_save, f)


    @classmethod
    def load(cls, path, config=None):
        r''' Load and return a CDB. This allows partial loads in probably not the right way at all.

        Args:
            path (`str`):
                Path to a `cdb.dat` from which to load data.
        '''
        with open(path, 'rb') as f:
            # Again no idea
            data = dill.load(f)
            if config is None:
                config = Config.from_dict(data['config'])
            # Create an instance of the CDB (empty)
            cdb = cls(config=config)

            # Load data into the new cdb instance
            for k in cdb.__dict__:
                if k in data['cdb']:
                    cdb.__dict__[k] = data['cdb'][k]

        return cdb


    def import_old_cdb_vectors(self, cdb):
        # Import context vectors
        for cui in self.cui2context_vectors:
            if cui in cdb.cui2context_vec:
                self.cui2context_vectors[cui] = {'medium': cdb.cui2context_vec[cui],
                                                 'long': cdb.cui2context_vec[cui],
                                                 'xlong': cdb.cui2context_vec[cui]}

                if cui in cdb.cui2context_vec_short:
                    self.cui2context_vectors[cui]['short'] = cdb.cui2context_vec_short[cui]

                self.cui2count_train[cui] = cdb.cui_count[cui]


    def import_old_cdb(self, cdb, import_vectors=True):
        r''' Import all data except for cuis and names from an old CDB.
        '''
        
        # Import vectors
        if import_vectors:
            self.import_old_cdb_vectors(cdb)

        # Import TUIs
        for cui in cdb.cui2names:
            self.cui2type_ids[cui] = {cdb.cui2tui.get(cui, 'unk')}

        # Import TUI to CUIs
        self.addl_info['type_id2cuis'] = cdb.tui2cuis

        # Import type_id to name
        self.addl_info['type_id2name'] = cdb.tui2name

        # Import description
        self.addl_info['cui2description'] = cdb.cui2desc

        # Import ICD10 and SNOMED
        self.addl_info['cui2snomed'] = {}
        for cui in self.cui2names:
            if cui in cdb.cui2info and 'icd10' in cdb.cui2info[cui]:
                self.addl_info['cui2icd10'][cui] = cdb.cui2info[cui]['icd10']
            if cui in cdb.cui2info and 'snomed' in cdb.cui2info[cui]:
                self.addl_info['cui2snomed'][cui] = cdb.cui2info[cui]['snomed']
            if cui in cdb.cui2info and 'opcs4' in cdb.cui2info[cui]:
                self.addl_info['cui2opcs4'][cui] = cdb.cui2info[cui]['opcs4']


        # Import cui 2 ontologies
        self.addl_info['cui2ontologies'] = cdb.cui2ontos


    def import_training(self, cdb, overwrite=True):
        r''' This will import vector embeddings from another CDB. No new concepts will be added.
        IMPORTANT it will not import name maps (cui2names, name2cuis or anything else) only vectors.

        Args:
            cdb (`medcat.cdb.CDB`):
                Concept database from which to import training vectors
            overwrite (`bool`, defaults to `True`):
                If True all training data in the existing CDB will be overwritten, else
                the average between the two training vectors will be taken.

        Examples:
            >>> new_cdb.import_traininig(cdb=old_cdb, owerwrite=True)
        '''
        # Import vectors and counts
        for cui in cdb.cui2context_vectors:
            if cui in self.cui2context_vectors:
                for context_type, vector in cdb.cui2context_vectors[cui].items():
                    if overwrite or context_type not in self.cdb.cui2context_vectors[cui]:
                        self.cui2context_vectors[cui][context_type] = vector
                    else:
                        self.cui2context_vectors[cui][context_type] = (vector + self.cui2context_vectors[cui][context_type]) / 2

                # Increase the vector count
                self.cui2count_train[cui] += cdb.cui2count_train[cui]


    def reset_cui_count(self, n=10):
        r''' Reset the CUI count for all concepts that received training, used when starting new unsupervised training
        or for suppervised with annealing.

        Args:
            n (`int`, optional, defaults to 10):
                This will be set as the CUI count for all cuis in this CDB.

        Examples:
            >>> cdb.reset_cui_count()
        '''
        for cui in self.cui2count_train.keys():
            self.cui2count_train[cui] = n


    def reset_training(self):
        r''' Will remove all training efforts - in other words all embeddings that are learnt
        for concepts in the current CDB. Please note that this does not remove synonyms (names) that were
        potentially added during supervised/online learning.
        '''
        self.reset_cui_count(n=0)
        for context_type in self.cui2context_vectors:
            self.cui2context_vectors[context_type] = {}
        self.reset_concept_similarity()


    def filter_by_cui(self, cuis_to_keep):
        ''' Subset the core CDB fields (dictionaries/maps). Note that this will potenitally keep a bit more CUIs
        then in cuis_to_keep. It will first find all names that link to the cuis_to_keep and then
        find all CUIs that link to those names and keep all of them.
        This also will not remove any data from cdb.addl_info - as this field can contain data of
        unknown structure.

        Args:
            cuis_to_keep (`List[str]`):
                CUIs that will be kept, the rest will be removed (not completely, look above).
        '''
        # First get all names/snames that should be kept based on this CUIs
        names_to_keep = set()
        snames_to_keep = set()
        for cui in cuis_to_keep:
            names_to_keep.update(self.cui2names[cui])
            snames_to_keep.update(self.cui2snames[cui])

        # Based on the names get also the indirect CUIs that have to be kept
        all_cuis_to_keep = set()
        for name in names_to_keep:
            all_cuis_to_keep.update(self.name2cuis[name])

        new_name2cuis = {}
        new_name2cuis2status = {}
        new_cui2names = {}
        new_cui2snames = {}
        new_cui2context_vectors = {}
        new_cui2count_train = {}
        new_cui2tags = {} # Used to add custom tags to CUIs
        new_cui2type_ids = {}
        new_cui2preferred_name = {}

        # Subset cui2<whatever>
        for cui in all_cuis_to_keep:
            new_cui2names[cui] = self.cui2names[cui]
            new_cui2snames[cui] = self.cui2snames[cui]
            new_cui2context_vectors[cui] = self.cui2context_vectors[cui]
            new_cui2count_train[cui] = self.cui2count_train[cui]
            new_cui2tags[cui] = self.cui2tags[cui]
            new_cui2type_ids[cui] = self.cui2type_ids[cui]
            new_cui2preferred_name[cui] = self.cui2preferred_name[cui]

        # Subset name2<whatever>
        for name in names_to_keep:
            new_name2cuis[name] = self.name2cuis[name]
            new_name2cuis2status[name] = self.name2cuis2status[name]

        # Replace everything
        self.name2cuis = new_name2cuis
        self.snames = snames_to_keep
        self.name2cuis2status = new_name2cuis2status
        self.cui2names = new_cui2names
        self.cui2snames = new_cui2snames
        self.cui2context_vectors = new_cui2context_vectors
        self.cui2count_train = new_cui2count_train
        self.cui2tags = new_cui2tags
        self.cui2type_ids = new_cui2type_ids
        self.cui2preferred_name = new_cui2preferred_name


    def print_stats(self):
        r'''Print basic statistics for the CDB.
        '''
        self.log.info("Number of concepts: {:,}".format(len(self.cui2names)))
        self.log.info("Number of names:    {:,}".format(len(self.name2cuis)))
        self.log.info("Number of concepts that received training: {:,}".format(len([cui for cui in self.cui2count_train if self.cui2count_train[cui] > 0])))
        self.log.info("Number of seen training examples in total: {:,}".format(sum(self.cui2count_train.values())))
        self.log.info("Average training examples per concept:     {:.1f}".format(np.average(
            [self.cui2count_train[cui] for cui in self.cui2count_train if self.cui2count_train[cui] > 0])))


    def reset_concept_similarity(self):
        r''' Reset concept similarity matrix.
        '''
        self.addl_info['similarity'] = {}


    def most_similar(self, cui, context_type, type_id_filter=[], min_cnt=0, topn=50, force_build=False):
        r''' Given a concept it will calculate what other concepts in this CDB have the most similar
        embedding.

        Args:
            cui (`str`):
                The concept ID for the base concept for which you want to get the most similar concepts.
            context_type (`str`):
                On what vector type from the cui2context_vectors map will the similarity be calculated.
            type_id_filter (`List[str]`):
                A list of type_ids that will be used to filterout the returned results. Using this it is possible
                to limit the similarity calculation to only disorders/symptoms/drugs/...
            min_cnt (`int`):
                Minimum training examples (unsupervised+supervised) that a concept must have to be considered
                for the similarity calculation.
            topn (`int`):
                How many results to return
            force_build (`bool`, defaults to `False`):
                Do not use cached sim matrix

        Return:
            results (Dict):
                A dictionary with topn results like: {<cui>: {'name': <name>, 'sim': <similarity>, 'type_name': <type_name>,
                                                              'type_id': <type_id>, 'cnt': <number of training examples the concept has seen>}, ...}

        '''

        if 'similarity' in self.addl_info:
            if context_type not in self.addl_info['similarity']:
                self.addl_info['similarity'][context_type] = {}
        else:
            self.addl_info['similarity'] = {context_type: {}}

        sim_data = self.addl_info['similarity'][context_type]

        # Create the matrix if necessary
        if 'sim_vectors' not in sim_data or force_build:
            self.log.info("Building similarity matrix")

            sim_vectors = []
            sim_vectors_counts = []
            sim_vectors_type_ids = []
            sim_vectors_cuis = []
            for _cui in self.cui2context_vectors:
                if context_type in self.cui2context_vectors[_cui]:
                    sim_vectors.append(unitvec(self.cui2context_vectors[_cui][context_type]))
                    sim_vectors_counts.append(self.cui2count_train[_cui])
                    sim_vectors_type_ids.append(self.cui2type_ids.get(_cui, {'unk'}))
                    sim_vectors_cuis.append(_cui)

            sim_data['sim_vectors'] = np.array(sim_vectors)
            sim_data['sim_vectors_counts'] = np.array(sim_vectors_counts)
            sim_data['sim_vectors_type_ids'] = np.array(sim_vectors_type_ids)
            sim_data['sim_vectors_cuis'] = np.array(sim_vectors_cuis)

        # Select appropriate concepts
        type_id_inds = np.arange(0, len(sim_data['sim_vectors_type_ids']))
        if len(type_id_filter) > 0:
            type_id_inds = np.array([], dtype=np.int32)
            for type_id in type_id_filter:
                type_id_inds = np.union1d(np.array([ind for ind, type_ids in enumerate(sim_data['sim_vectors_type_ids']) if type_id in type_ids]),
                        type_id_inds)
        cnt_inds = np.arange(0, len(sim_data['sim_vectors_counts']))
        if min_cnt > 0:
            cnt_inds = np.where(sim_data['sim_vectors_counts'] >= min_cnt)[0]
        # Intersect cnt and type_id 
        inds = np.intersect1d(type_id_inds, cnt_inds)

        mtrx = sim_data['sim_vectors'][inds]
        cuis = sim_data['sim_vectors_cuis'][inds]

        sims = np.dot(mtrx, unitvec(self.cui2context_vectors[cui][context_type]))

        sims_srt = np.argsort(-1*sims)

        # Create the return dict
        res = {}
        print()
        for ind, _cui in enumerate(cuis[sims_srt[0:topn]]):
            res[_cui] = {'name': self.cui2preferred_name.get(_cui, list(self.cui2names[_cui])[0]), 'sim': sims[sims_srt][ind],
                         'type_names': [self.addl_info['type_id2name'].get(cui, 'unk') for cui in self.cui2type_ids.get(_cui, ['unk'])],
                         'type_ids': self.cui2type_ids.get(_cui, 'unk'),
                         'cnt': self.cui2count_train[_cui]}

        return res
