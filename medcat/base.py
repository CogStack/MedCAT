from abc import ABC, abstractmethod
from typing import Optional, Iterable, Dict, List, Union, Tuple, Set
import os

from spacy.tokens import Doc

from medcat.utils.checkpoint import Checkpoint
from medcat.config import Config
from medcat.cdbbase import CDBBase


class CATBase(ABC):
    DEFAULT_MODEL_PACK_NAME = "medcat_model_pack"

    def __init__(self) -> None:
        self.cdb: CDBBase
        self.config: Config

    @abstractmethod
    def get_hash(self, force_recalc: bool = False) -> str:
        """Will not be a deep hash but will try to catch all the changing parts during training.

        Able to force recalculation of hash. This is relevant for CDB
        the hash for which is otherwise only recalculated if it has changed.

        Args:
            force_recalc (bool, optional): Whether to force recalculation. Defaults to False.

        Returns:
            str: The resulting hash
        """
    
    @abstractmethod
    def get_model_card(self, as_dict: bool = False):
        """A minimal model card for MedCAT model packs.

        Args:
            as_dict (bool):
                Whether to return the model card as a dictionary instead of a str (Default value False).

        Returns:
            str:
                The string representation of the JSON object.
            OR
            dict:
                The dict JSON object.
        """

    @abstractmethod
    def create_model_pack(self, save_dir_path: str, model_pack_name: str = DEFAULT_MODEL_PACK_NAME, force_rehash: bool = False,
            cdb_format: str = 'dill') -> str:
        """Will crete a .zip file containing all the models in the current running instance
        of MedCAT. This is not the most efficient way, for sure, but good enough for now.

        Args:
            save_dir_path (str):
                An id will be appended to this name
            model_pack_name (str, optional):
                The model pack name. Defaults to DEFAULT_MODEL_PACK_NAME.
            force_rehash (bool, optional):
                Force recalculation of hash. Defaults to `False`.
            cdb_format (str):
                The format of the saved CDB in the model pack.
                The available formats are:
                - dill
                - json
                Defaults to 'dill'

        Returns:
            str:
                Model pack name
        """
    
    @abstractmethod
    def __call__(self, text: Optional[str], do_train: bool = False) -> Optional[Doc]:
        """Push the text through the pipeline.

        Args:
            text (Optional[str]):
                The text to be annotated, if the text length is longer than
                self.config.preprocessing['max_document_length'] it will be trimmed to that length.
            do_train (bool):
                This causes so many screwups when not there, so I'll force training
                to False. To run training it is much better to use the self.train() function
                but for some special cases I'm leaving it here also.
                Defaults to `False`.
        Returns:
            Optional[Doc]:
                A single spacy document or multiple spacy documents with the extracted entities
        """

    @abstractmethod
    def train(self,
              data_iterator: Iterable,
              nepochs: int = 1,
              fine_tune: bool = True,
              progress_print: int = 1000,
              checkpoint: Optional[Checkpoint] = None,
              is_resumed: bool = False) -> None:
        """Runs training on the data, note that the maximum length of a line
        or document is 1M characters. Anything longer will be trimmed.

        Args:
            data_iterator (Iterable):
                Simple iterator over sentences/documents, e.g. a open file
                or an array or anything that we can use in a for loop.
            nepochs (int):
                Number of epochs for which to run the training.
            fine_tune (bool):
                If False old training will be removed.
            progress_print (int):
                Print progress after N lines.
            checkpoint (Optional[medcat.utils.checkpoint.CheckpointUT]):
                The MedCAT checkpoint object
            is_resumed (bool):
                If True resume the previous training; If False, start a fresh new training.
        """

    @abstractmethod
    def train_supervised_raw(self,
                             data: Dict[str, List[Dict[str, dict]]],
                             reset_cui_count: bool = False,
                             nepochs: int = 1,
                             print_stats: int = 0,
                             use_filters: bool = False,
                             terminate_last: bool = False,
                             use_overlaps: bool = False,
                             use_cui_doc_limit: bool = False,
                             test_size: int = 0,
                             devalue_others: bool = False,
                             use_groups: bool = False,
                             never_terminate: bool = False,
                             train_from_false_positives: bool = False,
                             extra_cui_filter: Optional[Set] = None,
                             retain_extra_cui_filter: bool = False,
                             checkpoint: Optional[Checkpoint] = None,
                             retain_filters: bool = False,
                             is_resumed: bool = False) -> Tuple:
        """Train supervised based on the raw data provided.

        The raw data is expected in the following format:
        {'projects':
            [ # list of projects
                { # project 1
                    'name': '<some name>',
                    # list of documents
                    'documents': [{'name': '<some name>',  # document 1
                                    'text': '<text of the document>',
                                    # list of annotations
                                    'annotations': [{'start': -1,  # annotation 1
                                                    'end': 1,
                                                    'cui': 'cui',
                                                    'value': '<text value>'}, ...],
                                    }, ...]
                }, ...
            ]
        }

        Please take care that this is more a simulated online training then supervised.

        When filtering, the filters within the CAT model are used first,
        then the ones from MedCATtrainer (MCT) export filters,
        and finally the extra_cui_filter (if set).
        That is to say, the expectation is:
        extra_cui_filter ⊆ MCT filter ⊆ Model/config filter.

        Args:
            data (Dict[str, List[Dict[str, dict]]]):
                The raw data, e.g from MedCATtrainer on export.
            reset_cui_count (boolean):
                Used for training with weight_decay (annealing). Each concept has a count that is there
                from the beginning of the CDB, that count is used for annealing. Resetting the count will
                significantly increase the training impact. This will reset the count only for concepts
                that exist in the the training data.
            nepochs (int):
                Number of epochs for which to run the training.
            print_stats (int):
                If > 0 it will print stats every print_stats epochs.
            use_filters (boolean):
                Each project in medcattrainer can have filters, do we want to respect those filters
                when calculating metrics.
            terminate_last (boolean):
                If true, concept termination will be done after all training.
            use_overlaps (boolean):
                Allow overlapping entities, nearly always False as it is very difficult to annotate overlapping entities.
            use_cui_doc_limit (boolean):
                If True the metrics for a CUI will be only calculated if that CUI appears in a document, in other words
                if the document was annotated for that CUI. Useful in very specific situations when during the annotation
                process the set of CUIs changed.
            test_size (float):
                If > 0 the data set will be split into train test based on this ration. Should be between 0 and 1.
                Usually 0.1 is fine.
            devalue_others(bool):
                Check add_name for more details.
            use_groups (boolean):
                If True concepts that have groups will be combined and stats will be reported on groups.
            never_terminate (boolean):
                If True no termination will be applied
            train_from_false_positives (boolean):
                If True it will use false positive examples detected by medcat and train from them as negative examples.
            extra_cui_filter(Optional[Set]):
                This filter will be intersected with all other filters, or if all others are not set then only this one will be used.
            retain_extra_cui_filter(bool):
                Whether to retain the extra filters instead of the MedCATtrainer export filters.
                This will only have an effect if/when retain_filters is set to True. Defaults to False.
            checkpoint (Optional[Optional[medcat.utils.checkpoint.CheckpointST]):
                The MedCAT CheckpointST object
            retain_filters (bool):
                If True, retain the filters in the MedCATtrainer export within this CAT instance. In other words, the
                filters defined in the input file will henseforth be saved within config.linking.filters .
                This only makes sense if there is only one project in the input data. If that is not the case,
                a ValueError is raised. The merging is done in the first epoch.
            is_resumed (bool):
                If True resume the previous training; If False, start a fresh new training.
        Returns:
            fp (dict):
                False positives for each CUI.
            fn (dict):
                False negatives for each CUI.
            tp (dict):
                True positives for each CUI.
            p (dict):
                Precision for each CUI.
            r (dict):
                Recall for each CUI.
            f1 (dict):
                F1 for each CUI.
            cui_counts (dict):
                Number of occurrence for each CUI.
            examples (dict):
                FP/FN examples of sentences for each CUI.
        """

    @abstractmethod
    def get_entities(self,
                     text: str,
                     only_cui: bool = False,
                     addl_info: List[str] = ['cui2icd10', 'cui2ontologies', 'cui2snomed']) -> Dict:
        doc = self(text)
        out = self._doc_to_out(doc, only_cui, addl_info)  # type: ignore
        return out

    @abstractmethod
    def get_entities_multi_texts(self,
                     texts: Union[Iterable[str], Iterable[Tuple]],
                     only_cui: bool = False,
                     addl_info: List[str] = ['cui2icd10', 'cui2ontologies', 'cui2snomed'],
                     n_process: Optional[int] = None,
                     batch_size: Optional[int] = None) -> List[Dict]:
        """Get entities

        Args:
            texts (Union[Iterable[str], Iterable[Tuple]]): Text to be annotated
            only_cui (bool, optional): Whether to only return CUIs. Defaults to False.
            addl_info (List[str], optional): Additional info. Defaults to ['cui2icd10', 'cui2ontologies', 'cui2snomed'].
            n_process (Optional[int], optional): Number of processes. Defaults to None.
            batch_size (Optional[int], optional): The size of a batch. Defaults to None.

        Returns:
            List[Dict]: List of entity documents.
        """

    @abstractmethod
    def get_json(self, text: str, only_cui: bool = False, addl_info=['cui2icd10', 'cui2ontologies']) -> str:
        """Get output in json format

        Args:
            text (str): Text to be annotated
            only_cui (bool, optional): Whether to only get CUIs. Defaults to False.
            addl_info (list, optional): Additional info. Defaults to ['cui2icd10', 'cui2ontologies'].

        Returns:
            str: Json with fields {'entities': <>, 'text': text}.
        """
    
    @abstractmethod
    def multiprocessing(self,
                        data: Union[List[Tuple], Iterable[Tuple]],
                        nproc: int = 2,
                        batch_size_chars: int = 5000 * 1000,
                        only_cui: bool = False,
                        addl_info: List[str] = [],
                        separate_nn_components: bool = True,
                        out_split_size_chars: Optional[int] = None,
                        save_dir_path: str = os.path.abspath(os.getcwd()),
                        min_free_memory=0.1) -> Dict:
        r"""Run multiprocessing for inference, if out_save_path and out_split_size_chars is used this will also continue annotating
        documents if something is saved in that directory.

        Args:
            data:
                Iterator or array with format: [(id, text), (id, text), ...]
            nproc (int):
                Number of processors. Defaults to 8.
            batch_size_chars (int):
                Size of a batch in number of characters, this should be around: NPROC * average_document_length * 200.
                Defaults to 1000000.
            separate_nn_components (bool):
                If set the medcat pipe will be broken up into NN and not-NN components and
                they will be run sequentially. This is useful as the NN components
                have batching and like to process many docs at once, while the rest of the pipeline
                runs the documents one by one. Defaults to True.
            out_split_size_chars (Optional[int]):
                If set once more than out_split_size_chars are annotated
                they will be saved to a file (save_dir_path) and the memory cleared. Recommended
                value is 20*batch_size_chars.
            save_dir_path(str):
                Where to save the annotated documents if splitting. Defaults to the current working directory.
            min_free_memory(float):
                If set a process will not start unless there is at least this much RAM memory left,
                should be a range between [0, 1] meaning how much of the memory has to be free. Helps when annotating
                very large datasets because spacy is not the best with memory management and multiprocessing.
                Defaults to 0.1.

        Returns:
            Dict:
                {id: doc_json, id2: doc_json2, ...}, in case out_split_size_chars is used
                the last batch will be returned while that and all previous batches will be
                written to disk (out_save_dir).
        """

    @abstractmethod
    def multiprocessing_pipe(self,
                             in_data: Union[List[Tuple], Iterable[Tuple]],
                             nproc: Optional[int] = None,
                             batch_size: Optional[int] = None,
                             only_cui: bool = False,
                             addl_info: List[str] = [],
                             return_dict: bool = True,
                             batch_factor: int = 2) -> Union[List[Tuple], Dict]:
        """Run multiprocessing NOT FOR TRAINING

        Args:
            in_data (Union[List[Tuple], Iterable[Tuple]]): List with format: [(id, text), (id, text), ...]
            nproc (Optional[int], optional): The number of processors. Defaults to None.
            batch_size (Optional[int], optional): The number of texts to buffer. Defaults to None.
            only_cui (bool, optional): Whether to get only CUIs. Defaults to False.
            addl_info (List[str], optional): Additional info. Defaults to [].
            return_dict (bool, optional): Flag for returning either a dict or a list of tuples. Defaults to True.
            batch_factor (int, optional): Batch factor. Defaults to 2.

        Raises:
            ValueError:
                When number of processes is 0.

        Returns:
            Union[List[Tuple], Dict]:
                {id: doc_json, id: doc_json, ...} or if return_dict is False, a list of tuples: [(id, doc_json), (id, doc_json), ...]
        """