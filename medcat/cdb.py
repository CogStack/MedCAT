"""Representation class for CDB data
"""
import dill
import json
import logging
import aiofiles
import numpy as np
from typing import Dict, Set, Optional, List, Union
from functools import partial

from medcat import __version__
from medcat.utils.hasher import Hasher
from medcat.utils.matutils import unitvec
from medcat.utils.ml_utils import get_lr_linking
from medcat.config import Config, weighted_average, workers
from medcat.utils.saving.serializer import CDBSerializer


logger = logging.getLogger(__name__)


class CDB(object):
    """Concept DataBase - holds all information necessary for NER+L.

    Properties:
        name2cuis (Dict[str, List[str]]):
            Map fro concept name to CUIs - one name can map to multiple CUIs.
        name2cuis2status (Dict[str, Dict[str, str]]):
            What is the status for a given name and cui pair - each name can be:
                P - Preferred, A - Automatic (e.g. let medcat decide), N - Not common.
        snames (Set[str]):
            All possible subnames for all concepts
        cui2names (Dict[str, Set[str]]):
            From cui to all names assigned to it. Mainly used for subsetting (maybe even only).
        cui2snames (Dict[str, Set[str]]):
            From cui to all sub-names assigned to it. Only used for subsetting.
        cui2context_vectors (Dict[str, Dict[str, np.array]]):
            From cui to a dictionary of different kinds of context vectors. Normally you would have here
            a short and a long context vector - they are calculated separately.
        cui2count_train (Dict[str, int]):
            From CUI to the number of training examples seen.
        cui2tags (Dict[str, List[str]]):
            From CUI to a list of tags. This can be used to tag concepts for grouping of whatever.
        cui2type_ids (Dict[str, Set[str]]):
            From CUI to type id (e.g. TUI in UMLS).
        cui2preferred_name (Dict[str, str]):
            From CUI to the preferred name for this concept.
        cui2average_confidence (Dict[str, str]):
            Used for dynamic thresholding. Holds the average confidence for this CUI given the training examples.
        name2count_train (Dict[str, str]):
            Counts how often did a name appear during training.
        addl_info (Dict[str, Dict[]]):
            Any additional maps that are not part of the core CDB. These are usually not needed
            for the base NER+L use-case, but can be useufl for Debugging or some special stuff.
        vocab (Dict[str, int]):
            Stores all the words tha appear in this CDB and the count for each one.
        is_dirty (bool):
            Whether or not the CDB has been changed since it was loaded or created
    """

    def __init__(self, config: Union[Config, None] = None) -> None:
        if config is None:
            self.config = Config()
        else:
            self.config = config
        self.name2cuis: Dict = {}
        self.name2cuis2status: Dict = {}

        self.snames: Set = set()

        self.cui2names: Dict = {}
        self.cui2snames: Dict = {}
        self.cui2context_vectors: Dict = {}
        self.cui2count_train: Dict = {}
        self.cui2info: Dict = {}
        self.cui2tags: Dict = {} # Used to add custom tags to CUIs
        self.cui2type_ids: Dict = {}
        self.cui2preferred_name: Dict = {}
        self.cui2average_confidence: Dict = {}
        self.name2count_train: Dict = {}
        self.name_isupper: Dict = {}

        self.addl_info: Dict= {
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
        self.vocab: Dict = {} # Vocabulary of all words ever in our cdb
        self._optim_params = None
        self.is_dirty = False
        self._hash: Optional[str] = None
        self._memory_optimised_parts: Set[str] = set()

    def get_name(self, cui: str) -> str:
        """Returns preferred name if it exists, otherwise it will return
        the longest name assigned to the concept.

        Args:
            cui
        """
        name = cui # In case we do not find anything it will just return the CUI

        if cui in self.cui2preferred_name and self.cui2preferred_name[cui]:
            name = self.cui2preferred_name[cui]
        elif cui in self.cui2names and self.cui2names[cui]:
            name = " ".join(str(max(self.cui2names[cui], key=len)).split(self.config.general.get('separator', '~'))).title()

        return name

    def update_cui2average_confidence(self, cui: str, new_sim: float) -> None:
        self.cui2average_confidence[cui] = (self.cui2average_confidence.get(cui, 0) * self.cui2count_train.get(cui, 0) + new_sim) / \
                                            (self.cui2count_train.get(cui, 0) + 1)
        self.is_dirty = True

    def remove_names(self, cui: str, names: Dict) -> None:
        """Remove names from an existing concept - effect is this name will never again be used to link to this concept.
        This will only remove the name from the linker (namely name2cuis and name2cuis2status), the name will still be present everywhere else.
        Why? Because it is bothersome to remove it from everywhere, but
        could also be useful to keep the removed names in e.g. cui2names.

        Args:
            cui (str):
                Concept ID or unique identifer in this database.
            names (Dict[str, Dict]):
                Names to be removed, should look like: `{'name': {'tokens': tokens, 'snames': snames, 'raw_name': raw_name}, ...}`
        """
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
        self.is_dirty = True

    def remove_cui(self, cui: str) -> None:
        """This function takes a `CUI` as an argument and removes it from all the internal objects that reference it.
        Args:
            cui
        """
        if cui in self.cui2names:
            del self.cui2names[cui]
        if cui in self.cui2snames:
            del self.cui2snames[cui]
        if cui in self.cui2context_vectors:
            del self.cui2context_vectors[cui]
        if cui in self.cui2count_train:
            del self.cui2count_train[cui]
        if cui in self.cui2tags:
            del self.cui2tags[cui]
        if cui in self.cui2type_ids:
            del self.cui2type_ids[cui]
        if cui in self.cui2preferred_name:
            del self.cui2preferred_name[cui]
        if cui in self.cui2average_confidence:
            del self.cui2average_confidence[cui]
        for name, cuis in self.name2cuis.items():
            if cui in cuis:
                cuis.remove(cui)
        for name, cuis2status in self.name2cuis2status.items():
            if cui in cuis2status:
                del cuis2status[cui]
        if isinstance(self.snames, set):
            # if this is a memory optimised CDB, this won't be a set
            # but it also won't need to be changed since it
            # relies directly on cui2snames
            self.snames = set()
            for cuis in self.cui2snames.values():
                self.snames |= cuis
        self.name2count_train = {name: len(cuis) for name, cuis in self.name2cuis.items()}
        self.is_dirty = True

    def add_names(self, cui: str, names: Dict, name_status: str = 'A', full_build: bool = False) -> None:
        """Adds a name to an existing concept.

        Args:
            cui (str):
                Concept ID or unique identifer in this database, all concepts that have
                the same CUI will be merged internally.
            names (Dict[str, Dict]):
                Names for this concept, or the value that if found in free text can be linked to this concept.
                Names is an dict like: `{name: {'tokens': tokens, 'snames': snames, 'raw_name': raw_name}, ...}`
            name_status (str):
                One of `P`, `N`, `A`
            full_build (bool)):
                If True the dictionary self.addl_info will also be populated, contains a lot of extra information
                about concepts, but can be very memory consuming. This is not necessary
                for normal functioning of MedCAT (Default value `False`).
        """
        name_status = name_status.upper()
        if name_status not in ['P', 'A', 'N']:
            # Name status must be one of the three
            name_status = 'A'

        self.add_concept(cui=cui, names=names, ontologies=set(), name_status=name_status, type_ids=set(), description='', full_build=full_build)

    def add_concept(self,
                    cui: str,
                    names: Dict,
                    ontologies: set,
                    name_status: str,
                    type_ids: Set[str],
                    description: str,
                    full_build: bool = False) -> None:
        """Add a concept to internal Concept Database (CDB). Depending on what you are providing
        this will add a large number of properties for each concept.

        Args:
            cui (str):
                Concept ID or unique identifier in this database, all concepts that have
                the same CUI will be merged internally.
            names (Dict[str, Dict]):
                Names for this concept, or the value that if found in free text can be linked to this concept.
                Names is an dict like: `{name: {'tokens': tokens, 'snames': snames, 'raw_name': raw_name}, ...}`
            ontologies (Set[str]):
                ontologies in which the concept exists (e.g. SNOMEDCT, HPO)
            name_status (str):
                One of `P`, `N`, `A`
            type_ids (Set[str]):
                Semantic type identifier (have a look at TUIs in UMLS or SNOMED-CT)
            description (str):
                Description of this concept.
            full_build (bool):
                If True the dictionary self.addl_info will also be populated, contains a lot of extra information
                about concepts, but can be very memory consuming. This is not necessary
                for normal functioning of MedCAT (Default Value `False`).
        """
        # Add CUI to the required dictionaries
        if cui not in self.cui2names:
            # Create placeholders
            self.cui2names[cui] = set()
            self.cui2snames[cui] = set()

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
            # Extend cui2snames, but check is the cui already in also
            if cui in self.cui2snames:
                self.cui2snames[cui].update(name_info['snames'])
            else:
                self.cui2snames[cui] = name_info['snames']

            # Add whether concept is uppercase
            self.name_isupper[name] = names[name]['is_upper']

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
                if ontologies:
                    self.addl_info['cui2ontologies'][cui] = ontologies
                if description:
                    self.addl_info['cui2description'][cui] = description
                self.addl_info['cui2original_names'][cui] = set([v['raw_name'] for k, v in names.items()])
            else:
                # Update existing ones
                if ontologies:
                    self.addl_info['cui2ontologies'][cui].update(ontologies)
                if description:
                    self.addl_info['cui2description'][cui] = description
                self.addl_info['cui2original_names'][cui].update([v['raw_name'] for k,v in names.items()])

            for type_id in type_ids:
                # Add type_id2cuis link
                if type_id in self.addl_info['type_id2cuis']:
                    self.addl_info['type_id2cuis'][type_id].add(cui)
                else:
                    self.addl_info['type_id2cuis'][type_id] = {cui}
        self.is_dirty = True

    def add_addl_info(self, name: str, data: Dict, reset_existing: bool = False) -> None:
        """Add data to the addl_info dictionary. This is done in a function to
        not directly access the addl_info dictionary.

        Args:
            name (str):
                What key should be used in the `addl_info` dictionary.
            data (Dict[<whatever>]):
                What will be added as the value for the key `name`
            reset_existing (bool):
                Should old data be removed if it exists
        """
        if reset_existing:
            self.addl_info[name] = {}

        self.addl_info[name].update(data)
        self.is_dirty = True

    def update_context_vector(self,
                              cui: str,
                              vectors: Dict[str, np.ndarray],
                              negative: bool = False,
                              lr: Optional[float] = None,
                              cui_count: int = 0) -> None:
        """Add the vector representation of a context for this CUI.

        cui (str):
            The concept in question.
        vectors (Dict[str, numpy.ndarray]):
            Vector represenation of the context, must have the format: {'context_type': np.array(<vector>), ...}
            context_type - is usually one of: ['long', 'medium', 'short']
        negative (bool):
            Is this negative context of positive (Default Value `False`).
        lr (int):
            If set it will override the base value from the config file.
        cui_count (int):
            The learning rate will be calculated based on the count for the provided CUI + cui_count.
            Defaults to 0.
        """
        if cui not in self.cui2context_vectors:
            self.cui2context_vectors[cui] = {}
            self.cui2count_train[cui] = 0

        similarity = None
        for context_type, vector in vectors.items():
            # Get the right context
            if context_type in self.cui2context_vectors[cui]:
                cv = self.cui2context_vectors[cui][context_type]
                similarity = np.dot(unitvec(cv), unitvec(vector))

                # Get the learning rate if None
                if lr is None:
                    lr = get_lr_linking(self.config, self.cui2count_train[cui] + cui_count)

                if negative:
                    # Add negative context
                    b = max(0, similarity) * lr
                    self.cui2context_vectors[cui][context_type] = cv*(1-b) - vector*b
                else:
                    b = (1 - max(0, similarity)) * lr
                    self.cui2context_vectors[cui][context_type] = cv*(1-b) + vector*b

                # DEBUG
                logger.debug("Updated vector embedding.\n" +
                        "CUI: %s, Context Type: %s, Similarity: %.2f, Is Negative: %s, LR: %.5f, b: %.3f", cui, context_type,
                            similarity, negative, lr, b)
                cv = self.cui2context_vectors[cui][context_type]
                similarity_after = np.dot(unitvec(cv), unitvec(vector))
                logger.debug("Similarity before vs after: %.5f vs %.5f", similarity, similarity_after)
            else:
                if negative:
                    self.cui2context_vectors[cui][context_type] = -1 * vector
                else:
                    self.cui2context_vectors[cui][context_type] = vector

                # DEBUG
                logger.debug("Added new context type with vectors.\n" +
                        "CUI: %s, Context Type: %s, Is Negative: %s", cui, context_type, negative)

        if not negative:
            # Increase counter only for positive examples
            self.cui2count_train[cui] += 1
        self.is_dirty = True

    def save(self, path: str, json_path: Optional[str] = None, overwrite: bool = True,
            calc_hash_if_missing: bool = False) -> None:
        """Saves model to file (in fact it saves variables of this class).

        If a `json_path` is specified, the JSON serialization is used for some of the data.

        Args:
            path (str):
                Path to a file where the model will be saved
            json_path (Optional[str]):
                If specified, json serialisation is used. Defaults to None.
            overwrite (bool):
                Whether or not to overwrite existing file(s).
            calc_hash_if_missing (bool):
                Calculate the hash if it's missing. Defaults to `False`
        """
        if calc_hash_if_missing and not self._hash:
            # get instead of calculate so that the CDB is marked as not dirty if it was dirty
            self.get_hash()
        ser = CDBSerializer(path, json_path)
        ser.serialize(self, overwrite=overwrite)

    # TODO - add JSON serialization to async save
    async def save_async(self, path: str) -> None:
        """Async version of saving model to file (in fact it saves variables of this class).

        This method does not (currently) support the new JSON serialization.

        Args:
            path (str):
                Path to a file where the model will be saved
        """
        async with aiofiles.open(path, 'wb') as f:
            to_save = {
                'config': self.config.__dict__,
                'cdb': {k: v for k, v in self.__dict__.items() if k != 'config'}
            }
            await f.write(dill.dumps(to_save))

    @classmethod
    def load(cls, path: str, json_path: Optional[str] = None, config_dict: Optional[Dict] = None) -> "CDB":
        """Load and return a CDB. This allows partial loads in probably not the right way at all.

        If `json_path` is specified, the JSON serialization is assumed to be present.
        Otherwise, it is assumed not to be present.

        Args:
            path (str):
                Path to a `cdb.dat` from which to load data.
            json_path (str):
                Path to the JSON serialized folder
            config_dict:
                A dictionary that will be used to overwrite existing fields in the config of this CDB
        """
        ser = CDBSerializer(path, json_path)
        cdb = ser.deserialize(CDB)
        cls._check_medcat_version(cdb.config.asdict())
        cls._ensure_backward_compatibility(cdb.config)

        # Overwrite the config with new data
        if config_dict is not None:
            cdb.config.merge_config(config_dict)

        return cdb

    def import_training(self, cdb: "CDB", overwrite: bool = True) -> None:
        """This will import vector embeddings from another CDB. No new concepts will be added.
        IMPORTANT it will not import name maps (cui2names, name2cuis or anything else) only vectors.

        Args:
            cdb (medcat.cdb.CDB):
                Concept database from which to import training vectors
            overwrite (bool):
                If True all training data in the existing CDB will be overwritten, else
                the average between the two training vectors will be taken (Default value `True`).

        Examples:

            >>> new_cdb.import_traininig(cdb=old_cdb, owerwrite=True)
        """
        # Import vectors and counts
        for cui in cdb.cui2context_vectors:
            if cui in self.cui2names:
                for context_type, vector in cdb.cui2context_vectors[cui].items():
                    if overwrite or context_type not in self.cui2context_vectors[cui]:
                        self.cui2context_vectors[cui][context_type] = vector
                    else:
                        self.cui2context_vectors[cui][context_type] = (vector + self.cui2context_vectors[cui][context_type]) / 2

                # Increase the vector count
                self.cui2count_train[cui] = self.cui2count_train.get(cui, 0) + cdb.cui2count_train[cui]
        self.is_dirty = True

    def reset_cui_count(self, n: int = 10) -> None:
        """Reset the CUI count for all concepts that received training, used when starting new unsupervised training
        or for suppervised with annealing.

        Args:
            n (int):
                This will be set as the CUI count for all cuis in this CDB (Default value 10).

        Examples:

            >>> cdb.reset_cui_count()
        """
        for cui in self.cui2count_train.keys():
            self.cui2count_train[cui] = n
        self.is_dirty = True

    def reset_training(self) -> None:
        """Will remove all training efforts - in other words all embeddings that are learnt
        for concepts in the current CDB. Please note that this does not remove synonyms (names) that were
        potentially added during supervised/online learning.
        """
        self.cui2count_train = {}
        self.cui2context_vectors = {}
        self.reset_concept_similarity()
        self.is_dirty = True

    def populate_cui2snames(self, force: bool = True) -> None:
        """Populate the cui2snames dict if it's empty.

        If the dict is not empty and the population is not force,
        nothing will happen.

        For now, this method simply populates all the names form
        cui2names into cui2snames.

        Args:
            force (bool, optional): Whether to force the (re-)population. Defaults to True.
        """
        if not force and self.cui2snames:
            return
        self.cui2snames.clear() # in case forced re-population
        # run through cui2names
        # and create new sets so that they can be independently modified
        for cui, names in self.cui2names.items():
            self.cui2snames[cui] = set(names)  # new set
        self.is_dirty = True

    def filter_by_cui(self, cuis_to_keep: Union[List[str], Set[str]]) -> None:
        """Subset the core CDB fields (dictionaries/maps). Note that this will potenitally keep a bit more CUIs
        then in cuis_to_keep. It will first find all names that link to the cuis_to_keep and then
        find all CUIs that link to those names and keep all of them.
        This also will not remove any data from cdb.addl_info - as this field can contain data of
        unknown structure.

        As a side note, if the CDB has been memory-optimised, filtering will undo this memory optimisation.
        This is because the dicts being involved will be rewritten.
        However, the memory optimisation can be performed again afterwards.

        Args:
            cuis_to_keep (List[str]):
                CUIs that will be kept, the rest will be removed (not completely, look above).
        """

        if not self.cui2snames:
            raise Exception("This CDB does not support subsetting - most likely because it is a `small/medium` version of a CDB")

        # First get all names/snames that should be kept based on this CUIs
        names_to_keep = set()
        snames_to_keep = set()
        for cui in cuis_to_keep:
            names_to_keep.update(self.cui2names.get(cui, []))
            snames_to_keep.update(self.cui2snames.get(cui, []))

        # Based on the names get also the indirect CUIs that have to be kept
        all_cuis_to_keep = set()
        for name in names_to_keep:
            all_cuis_to_keep.update(self.name2cuis.get(name, []))

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
            if cui in self.cui2names:
                new_cui2names[cui] = self.cui2names[cui]
                new_cui2snames[cui] = self.cui2snames[cui]
                if cui in self.cui2context_vectors:
                    new_cui2context_vectors[cui] = self.cui2context_vectors[cui]
                    # We assume that it must have the cui2count_train if it has a vector
                    new_cui2count_train[cui] = self.cui2count_train[cui]
                if cui in self.cui2tags:
                    new_cui2tags[cui] = self.cui2tags[cui]
                new_cui2type_ids[cui] = self.cui2type_ids[cui]
                if cui in self.cui2preferred_name:
                    new_cui2preferred_name[cui] = self.cui2preferred_name[cui]

        # Subset name2<whatever>
        for name in names_to_keep:
            if name in self.name2cuis:
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
        self.is_dirty = True
        # reset memory optimisation state
        self._memory_optimised_parts.clear()

    def make_stats(self):
        stats = {}
        stats["Number of concepts"] = len(self.cui2names)
        stats["Number of names"] = len(self.name2cuis)
        stats["Number of concepts that received training"] = len([cui for cui in self.cui2count_train if self.cui2count_train[cui] > 0])
        stats["Number of seen training examples in total"] = sum(self.cui2count_train.values())
        positive_count_trains = [self.cui2count_train[cui] for cui in self.cui2count_train if self.cui2count_train[cui] > 0]
        stats["Average training examples per concept"] = np.average(positive_count_trains) if positive_count_trains else 0.0

        return stats

    def print_stats(self) -> None:
        """Print basic statistics for the CDB."""
        logger.info(json.dumps(self.make_stats(), indent=2))

    def reset_concept_similarity(self) -> None:
        """Reset concept similarity matrix."""
        self.addl_info['similarity'] = {}
        self.is_dirty = True

    def most_similar(self,
                     cui: str,
                     context_type: str,
                     type_id_filter: List[str] = [],
                     min_cnt: int = 0,
                     topn: int = 50,
                     force_build: bool = False) -> Dict:
        """Given a concept it will calculate what other concepts in this CDB have the most similar
        embedding.

        Args:
            cui (str):
                The concept ID for the base concept for which you want to get the most similar concepts.
            context_type (str):
                On what vector type from the cui2context_vectors map will the similarity be calculated.
            type_id_filter (List[str]):
                A list of type_ids that will be used to filterout the returned results. Using this it is possible
                to limit the similarity calculation to only disorders/symptoms/drugs/...
            min_cnt (int):
                Minimum training examples (unsupervised+supervised) that a concept must have to be considered
                for the similarity calculation.
            topn (int):
                How many results to return
            force_build (bool):
                Do not use cached sim matrix (Default value False)

        Returns:
            results (Dict):
                A dictionary with topn results like: {<cui>: {'name': <name>, 'sim': <similarity>, 'type_name': <type_name>,
                                                              'type_id': <type_id>, 'cnt': <number of training examples the concept has seen>}, ...}

        """

        if 'similarity' in self.addl_info:
            if context_type not in self.addl_info['similarity']:
                self.addl_info['similarity'][context_type] = {}
        else:
            self.addl_info['similarity'] = {context_type: {}}

        sim_data = self.addl_info['similarity'][context_type]

        # Create the matrix if necessary
        if 'sim_vectors' not in sim_data or force_build:
            logger.info("Building similarity matrix")

            sim_vectors = []
            sim_vectors_counts = []
            sim_vectors_type_ids = []
            sim_vectors_cuis = []
            for _cui in self.cui2context_vectors:
                if context_type in self.cui2context_vectors[_cui]:
                    sim_vectors.append(unitvec(self.cui2context_vectors[_cui][context_type]))
                    sim_vectors_counts.append(self.cui2count_train.get(_cui, 0))
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
        for ind, _cui in enumerate(cuis[sims_srt[0:topn]]):
            res[_cui] = {'name': self.cui2preferred_name.get(_cui, list(self.cui2names[_cui])[0]), 'sim': sims[sims_srt][ind],
                         'type_names': [self.addl_info['type_id2name'].get(cui, 'unk') for cui in self.cui2type_ids.get(_cui, ['unk'])],
                         'type_ids': self.cui2type_ids.get(_cui, 'unk'),
                         'cnt': self.cui2count_train.get(_cui, 0)}

        return res

    @staticmethod
    def _ensure_backward_compatibility(config: Config) -> None:
        # Hacky way of supporting old CDBs
        weighted_average_function = config.linking.weighted_average_function
        if callable(weighted_average_function) and getattr(weighted_average_function, "__name__", None) == "<lambda>":
            # the following type ignoring is for mypy because it is unable to detect the signature
            config.linking.weighted_average_function = partial(weighted_average, factor=0.0004) # type: ignore
        if config.general.workers is None:
            config.general.workers = workers()
        disabled_comps = config.general.spacy_disabled_components
        if 'tagger' in disabled_comps and 'lemmatizer' not in disabled_comps:
            config.general.spacy_disabled_components.append('lemmatizer')

    @classmethod
    def _check_medcat_version(cls, config_data: Dict) -> None:
        cdb_medcat_version = config_data.get('version', {}).get('medcat_version', None)
        if cdb_medcat_version is None:
            logger.warning('The CDB was exported by an unknown version of MedCAT.')
        elif __version__.split(".")[:1] != cdb_medcat_version.split(".")[:1]:
            logger.warning(
                f"""You have MedCAT version '{__version__}' installed while the CDB was exported by MedCAT version '{cdb_medcat_version}'.
Please reinstall MedCAT or download the compatible model."""
            )
        elif __version__.split(".")[:2] != cdb_medcat_version.split(".")[:2]:
            logger.warning(
                f"""You have MedCAT version '{__version__}' installed while the CDB was exported by MedCAT version '{cdb_medcat_version}',
which may or may not work. If you experience any compatibility issues, please reinstall MedCAT
or download the compatible model."""
            )

    def get_hash(self, force_recalc: bool = False):
        if not force_recalc and self._hash and not self.is_dirty:
            logger.info("Reusing old hash of CDB since the CDB has not changed: %s", self._hash)
            return self._hash
        self.is_dirty = False
        return self.calculate_hash()

    def calculate_hash(self):
        logger.info("Recalculating hash for CDB, this may take a while")
        hasher = Hasher()

        for k,v in self.__dict__.items():
            if k in ['cui2countext_vectors', 'name2cuis']:
                hasher.update(v, length=False)
            elif k in ['_hash', 'is_dirty']:
                # ignore _hash since if it previously didn't exist, the
                # new hash would be different when the value does exist
                # and ignore is_dirty so that we get the same hash as previously
                continue
            elif k != 'config':
                hasher.update(v, length=True)

        self._hash = hasher.hexdigest()
        logger.info("Found new CDB hash: %s", self._hash)
        return self._hash
