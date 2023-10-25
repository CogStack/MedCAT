from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union, Set

import numpy as np

from medcat.config import Config


class CDBBase(ABC):

    def __init__(self) -> None:
        self.config: Config
        self.name2cuis: Dict[str, List[str]]
        self.name2cuis2status: Dict[str, Dict]
        self.snames: Set
        self.cui2names: Dict[str, Set[str]]
        self.cui2snames: Dict[str, Set[str]]
        self.cui2context_vectors: Dict[str, Dict[str, np.array]]
        self.cui2count_train: Dict[str, int]
        self.cui2info: Dict
        self.cui2tags: Dict[str, List[str]]
        self.cui2type_ids: Dict[str, Set[str]]
        self.cui2preferred_name: Dict[str, str]
        self.cui2average_confidence: Dict[str, float]
        self.name2count_train: Dict[str, int]
        self.name_isupper: Dict[str, bool]
        self.addl_info: Dict[str, Dict]
        self.vocab: Dict[str, int]
        self.is_dirty: bool

    @abstractmethod
    def get_name(self, cui: str) -> str:
        """Returns preferred name if it exists, otherwise it will return
        the longest name assigned to the concept.

        Args:
            cui
        """

    @abstractmethod
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

    @abstractmethod
    def remove_cui(self, cui: str) -> None:
        """This function takes a `CUI` as an argument and removes it from all the internal objects that reference it.
        Args:
            cui
        """

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def import_training(self, cdb: "CDBBase", overwrite: bool = True) -> None:
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

    @abstractmethod
    def reset_cui_count(self, n: int = 10) -> None:
        """Reset the CUI count for all concepts that received training, used when starting new unsupervised training
        or for suppervised with annealing.

        Args:
            n (int):
                This will be set as the CUI count for all cuis in this CDB (Default value 10).

        Examples:

            >>> cdb.reset_cui_count()
        """

    @abstractmethod
    def reset_training(self) -> None:
        """Will remove all training efforts - in other words all embeddings that are learnt
        for concepts in the current CDB. Please note that this does not remove synonyms (names) that were
        potentially added during supervised/online learning.
        """

    @abstractmethod
    def populate_cui2snames(self, force: bool = True) -> None:
        """Populate the cui2snames dict if it's empty.

        If the dict is not empty and the population is not force,
        nothing will happen.

        For now, this method simply populates all the names form
        cui2names into cui2snames.

        Args:
            force (bool, optional): Whether to force the (re-)population. Defaults to True.
        """

    @abstractmethod
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

    @abstractmethod
    def print_stats(self) -> None:
        """Print basic statistics for the CDB."""

    @abstractmethod
    def reset_concept_similarity(self) -> None:
        """Reset concept similarity matrix."""

    @abstractmethod
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

    @abstractmethod
    def get_hash(self, force_recalc: bool = False) -> str:
        """Get the has hof this CDB.

        Args:
            force_recalc (bool, optional): Whether to force a relcualtion. Defaults to False.

        Returns:
            str: The hash for this CDB.
        """
