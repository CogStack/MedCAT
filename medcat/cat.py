import os
import glob
import shutil
import pickle
import json
import logging
import math
import time
import psutil
from time import sleep
from multiprocess import Process, Manager, cpu_count
from multiprocess.queues import Queue
from multiprocess.synchronize import Lock
from typing import Union, List, Tuple, Optional, Dict, Iterable, Set
from itertools import islice, chain, repeat
from datetime import date
from tqdm.autonotebook import tqdm, trange
from spacy.tokens import Span, Doc, Token
import humanfriendly

from medcat import __version__
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.pipe import Pipe
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.cdb import CDB
from medcat.utils.data_utils import make_mc_train_test, get_false_positives
from medcat.utils.normalizers import BasicSpellChecker
from medcat.utils.checkpoint import Checkpoint, CheckpointConfig, CheckpointManager
from medcat.utils.helpers import tkns_from_doc, get_important_config_parameters, has_new_spacy
from medcat.utils.hasher import Hasher
from medcat.ner.vocab_based_ner import NER
from medcat.linking.context_based_linker import Linker
from medcat.preprocessing.cleaners import prepare_name
from medcat.meta_cat import MetaCAT
from medcat.rel_cat import RelCAT
from medcat.utils.meta_cat.data_utils import json_to_fake_spacy
from medcat.config import Config
from medcat.vocab import Vocab
from medcat.ner.transformers_ner import TransformersNER
from medcat.utils.saving.serializer import SPECIALITY_NAMES, ONE2MANY
from medcat.utils.saving.envsnapshot import get_environment_info, ENV_SNAPSHOT_FILE_NAME
from medcat.stats.stats import get_stats
from medcat.utils.filters import set_project_filters
from medcat.utils.usage_monitoring import UsageMonitor


logger = logging.getLogger(__name__) # separate logger from the package-level one


HAS_NEW_SPACY = has_new_spacy()

MIN_GEN_LEN_FOR_WARN = 10_000


class CAT(object):
    """The main MedCAT class used to annotate documents, it is built on top of spaCy
    and works as a spaCy pipeline. Creates an instance of a spaCy pipeline that can
    be used as a spacy nlp model.

    Args:
        cdb (medcat.cdb.CDB):
            The concept database that will be used for NER+L
        config (medcat.config.Config):
            Global configuration for medcat
        vocab (medcat.vocab.Vocab, optional):
            Vocabulary used for vector embeddings and spelling. Default: None
        meta_cats (list of medcat.meta_cat.MetaCAT, optional):
            A list of models that will be applied sequentially on each
            detected annotation.
        rel_cats (list of medcat.rel_cat.RelCAT, optional)
            List of models applied sequentially on all detected annotations.

    Attributes (limited):
        cdb (medcat.cdb.CDB):
            Concept database used with this CAT instance, please do not assign
            this value directly.
        config (medcat.config.Config):
            The global configuration for medcat. Usually cdb.config will be used for this
            field. WILL BE REMOVED - TEMPORARY PLACEHOLDER
        vocab (medcat.utils.vocab.Vocab):
            The vocabulary object used with this instance, please do not assign
            this value directly.

    Examples:

        >>> cat = CAT(cdb, vocab)
        >>> spacy_doc = cat("Put some text here")
        >>> print(spacy_doc.ents) # Detected entities
    """
    DEFAULT_MODEL_PACK_NAME = "medcat_model_pack"

    def __init__(self,
                 cdb: CDB,
                 vocab: Union[Vocab, None] = None,
                 config: Optional[Config] = None,
                 meta_cats: List[MetaCAT] = [],
                 rel_cats: List[RelCAT] = [],
                 addl_ner: Union[TransformersNER, List[TransformersNER]] = []) -> None:
        self.cdb = cdb
        self.vocab = vocab
        if config is None:
            # Take config from the cdb
            self.config = cdb.config
        else:
            # Take the new config and assign it to the CDB also
            self.config = config
            self.cdb.config = config
        self._meta_cats = meta_cats
        self._rel_cats = rel_cats
        self._addl_ner = addl_ner if isinstance(addl_ner, list) else [addl_ner]
        self._create_pipeline(self.config)
        self.usage_monitor = UsageMonitor(self.config.version.id, self.config.general.usage_monitor)

    def _create_pipeline(self, config: Config):
        # Set log level
        logger.setLevel(config.general.log_level)

        # Build the pipeline
        self.pipe = Pipe(tokenizer=spacy_split_all, config=config)
        self.pipe.add_tagger(tagger=tag_skip_and_punct,
                             name='skip_and_punct',
                             additional_fields=['is_punct'])

        if self.vocab is not None:
            spell_checker = BasicSpellChecker(cdb_vocab=self.cdb.vocab, config=config, data_vocab=self.vocab)
            self.pipe.add_token_normalizer(spell_checker=spell_checker, config=config)

            # Add NER
            self.ner = NER(self.cdb, config)
            self.pipe.add_ner(self.ner)

            # Add LINKER
            self.linker = Linker(self.cdb, self.vocab, config)
            self.pipe.add_linker(self.linker)

        # Add addl_ner if they exist
        for ner in self._addl_ner:
            self.pipe.add_addl_ner(ner, ner.config.general.name)

        # Add meta_annotation classes if they exist
        for meta_cat in self._meta_cats:
            self.pipe.add_meta_cat(meta_cat, meta_cat.config.general.category_name)

        for rel_cat in self._rel_cats:
            self.pipe.add_rel_cat(rel_cat, "_".join(list(rel_cat.config.general["labels2idx"].keys())))

        # Set max document length
        self.pipe.spacy_nlp.max_length = config.preprocessing.max_document_length

    def get_hash(self, force_recalc: bool = False) -> str:
        """Will not be a deep hash but will try to catch all the changing parts during training.

        Able to force recalculation of hash. This is relevant for CDB
        the hash for which is otherwise only recalculated if it has changed.

        Args:
            force_recalc (bool): Whether to force recalculation. Defaults to False.

        Returns:
            str: The resulting hash
        """
        hasher = Hasher()
        if self.config.general.simple_hash:
            logger.info("Using simplified hashing that only takes into account the model card")
            hasher.update(self.get_model_card())
            return hasher.hexdigest()
        hasher.update(self.cdb.get_hash(force_recalc))

        hasher.update(self.config.get_hash())

        for mc in self._meta_cats:
            hasher.update(mc.get_hash())

        for trf in self._addl_ner:
            hasher.update(trf.get_hash())

        return hasher.hexdigest()

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
        card = {
                'Model ID': self.config.version.id,
                'Last Modified On': self.config.version.last_modified,
                'History (from least to most recent)': self.config.version.history,
                'Description': self.config.version.description,
                'Source Ontology': self.config.version.ontology,
                'Location': self.config.version.location,
                'MetaCAT models': self.config.version.meta_cats,
                'Basic CDB Stats': self.config.version.cdb_info,
                'Performance': self.config.version.performance,
                'Important Parameters (Partial view, all available in cat.config)': get_important_config_parameters(self.config),
                'MedCAT Version': self.config.version.medcat_version
                }

        if as_dict:
            return card
        else:
            return json.dumps(card, indent=2, sort_keys=False)

    def _versioning(self, force_rehash: bool = False):
        # Check version info and do not allow without it
        if self.config.version.description == 'No description':
            logger.warning("Please consider populating the version information [description, performance, location, ontology] in cat.config.version")

        # Fill the stuff automatically that is needed for versioning
        m = self.get_hash(force_recalc=force_rehash)
        version = self.config.version
        if version.id is None or m != version.id:
            if version.id is not None:
                version.history.append(version['id'])
            version.id = m
            version.last_modified = date.today().strftime("%d %B %Y")
            version.cdb_info = self.cdb.make_stats()
            version.meta_cats = [meta_cat.get_model_card(as_dict=True) for meta_cat in self._meta_cats]
            version.medcat_version = __version__
            logger.warning("Please consider updating [description, performance, location, ontology] in cat.config.version")

    def create_model_pack(self, save_dir_path: str, model_pack_name: str = DEFAULT_MODEL_PACK_NAME, force_rehash: bool = False,
            cdb_format: str = 'dill') -> str:
        """Will crete a .zip file containing all the models in the current running instance
        of MedCAT. This is not the most efficient way, for sure, but good enough for now.

        Args:
            save_dir_path (str):
                An id will be appended to this name
            model_pack_name (str):
                The model pack name. Defaults to DEFAULT_MODEL_PACK_NAME.
            force_rehash (bool):
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
        # Spacy model always should be just the name, but during loading it can be reset to path
        self.config.general.spacy_model = os.path.basename(self.config.general.spacy_model)
        # Versioning
        self._versioning(force_rehash)
        model_pack_name += "_{}".format(self.config.version.id)

        logger.warning("This will save all models into a zip file, can take some time and require quite a bit of disk space.")
        _save_dir_path = save_dir_path
        save_dir_path = os.path.join(save_dir_path, model_pack_name)

        # Check format
        if cdb_format.lower() == 'json':
            json_path = save_dir_path # in the same folder!
        else:
            json_path = None # use dill formatting
        logger.info('Saving model pack with CDB in %s format', cdb_format)

        # expand user path to make this work with '~'
        os.makedirs(os.path.expanduser(save_dir_path), exist_ok=True)

        # Save the used spacy model
        spacy_path = os.path.join(save_dir_path, self.config.general.spacy_model)
        if str(self.pipe.spacy_nlp._path) != spacy_path:
            # First remove if something is there
            shutil.rmtree(spacy_path, ignore_errors=True)
            shutil.copytree(str(self.pipe.spacy_nlp._path), spacy_path)

        # Save the CDB
        cdb_path = os.path.join(save_dir_path, "cdb.dat")
        self.cdb.save(cdb_path, json_path)

        # Save the config
        config_path = os.path.join(save_dir_path, "config.json")
        self.cdb.config.save(config_path)

        # Save the Vocab
        vocab_path = os.path.join(save_dir_path, "vocab.dat")
        if self.vocab is not None:
            # We will allow creation of modelpacks without vocabs
            self.vocab.save(vocab_path)

        # Save addl_ner
        for comp in self.pipe.spacy_nlp.components:
            if isinstance(comp[1], TransformersNER):
                trf_path = os.path.join(save_dir_path, "trf_" + comp[1].config.general.name)
                comp[1].save(trf_path)

        # Save all meta_cats
        for comp in self.pipe.spacy_nlp.components:
            if isinstance(comp[1], MetaCAT):
                name = comp[0]
                meta_path = os.path.join(save_dir_path, "meta_" + name)
                comp[1].save(meta_path)
            if isinstance(comp[1], RelCAT):
                name = comp[0]
                rel_path = os.path.join(save_dir_path, "rel_" + name)
                comp[1].save(rel_path)

        # Add a model card also, why not
        model_card_path = os.path.join(save_dir_path, "model_card.json")
        with open(model_card_path, 'w') as f:
            json.dump(self.get_model_card(as_dict=True), f, indent=2)

        # add a dependency snapshot
        env_info = get_environment_info()
        env_info_path = os.path.join(save_dir_path, ENV_SNAPSHOT_FILE_NAME)
        with open(env_info_path, 'w') as f:
            json.dump(env_info, f)

        # Zip everything
        shutil.make_archive(os.path.join(_save_dir_path, model_pack_name), 'zip', root_dir=save_dir_path)

        # Log model card and return new name
        logger.info(self.get_model_card()) # Print the model card
        return model_pack_name

    @classmethod
    def attempt_unpack(cls, zip_path: str) -> str:
        """Attempt unpack the zip to a folder and get the model pack path.

        If the folder already exists, no unpacking is done.

        Args:
            zip_path (str): The ZIP path

        Returns:
            str: The model pack path
        """
        base_dir = os.path.dirname(zip_path)
        filename = os.path.basename(zip_path)

        foldername = filename.replace(".zip", '')

        model_pack_path = os.path.join(base_dir, foldername)
        if os.path.exists(model_pack_path):
            logger.info("Found an existing unzipped model pack at: {}, the provided zip will not be touched.".format(model_pack_path))
        else:
            logger.info("Unziping the model pack and loading models.")
            shutil.unpack_archive(zip_path, extract_dir=model_pack_path)
        return model_pack_path

    @classmethod
    def load_model_pack(cls,
                        zip_path: str,
                        meta_cat_config_dict: Optional[Dict] = None,
                        ner_config_dict: Optional[Dict] = None,
                        medcat_config_dict: Optional[Dict] = None,
                        load_meta_models: bool = True,
                        load_addl_ner: bool = True,
                        load_rel_models: bool = True) -> "CAT":
        """Load everything within the 'model pack', i.e. the CDB, config, vocab and any MetaCAT models
        (if present)

        Args:
            zip_path (str):
                The path to model pack zip.
            meta_cat_config_dict (Optional[Dict]):
                A config dict that will overwrite existing configs in meta_cat.
                e.g. meta_cat_config_dict = {'general': {'device': 'cpu'}}.
                Defaults to None.
            ner_config_dict (Optional[Dict]):
                A config dict that will overwrite existing configs in transformers ner.
                e.g. ner_config_dict = {'general': {'chunking_overlap_window': 6}.
                Defaults to None.
            medcat_config_dict (Optional[Dict]):
                A config dict that will overwrite existing configs in the main medcat config
                before pipe initialisation. This can be useful if wanting to change something
                that only takes effect at init time (e.g spacy model). Defaults to None.
            load_meta_models (bool):
                Whether to load MetaCAT models if present (Default value True).
            load_addl_ner (bool):
                Whether to load additional NER models if present (Default value True).
            load_rel_models (bool):
                Whether to load RelCAT models if present (Default value True).

        Returns:
            CAT: The resulting CAT object.
        """
        from medcat.cdb import CDB
        from medcat.vocab import Vocab
        from medcat.meta_cat import MetaCAT
        from medcat.rel_cat import RelCAT

        model_pack_path = cls.attempt_unpack(zip_path)

        # Load the CDB
        cdb: CDB = cls.load_cdb(model_pack_path)

        # load config
        config_path = os.path.join(model_pack_path, "config.json")
        cdb.load_config(config_path, medcat_config_dict)

        # TODO load addl_ner

        # Modify the config to contain full path to spacy model
        cdb.config.general.spacy_model = os.path.join(model_pack_path, os.path.basename(cdb.config.general.spacy_model))

        # Load Vocab
        vocab_path = os.path.join(model_pack_path, "vocab.dat")
        if os.path.exists(vocab_path):
            vocab = Vocab.load(vocab_path)
        else:
            vocab = None

        # Find ner models in the model_pack
        trf_paths = [os.path.join(model_pack_path, path) for path in os.listdir(model_pack_path) if path.startswith('trf_')] if load_addl_ner else []
        addl_ner = []
        for trf_path in trf_paths:
            trf = TransformersNER.load(save_dir_path=trf_path,config_dict=ner_config_dict)
            trf.cdb = cdb # Set the cat.cdb to be the CDB of the TRF model
            addl_ner.append(trf)

        # Find metacat models in the model_pack
        meta_cats: List[MetaCAT] = []
        if load_meta_models:
            meta_cats = [mc[1] for mc in cls.load_meta_cats(model_pack_path, meta_cat_config_dict)]

        # Find Rel models in model_pack
        rel_paths = [os.path.join(model_pack_path, path) for path in os.listdir(model_pack_path) if path.startswith('rel_')] if load_rel_models else []
        rel_cats = []
        for rel_path in rel_paths:
            rel_cats.append(RelCAT.load(load_path=rel_path))

        cat = cls(cdb=cdb, config=cdb.config, vocab=vocab, meta_cats=meta_cats, addl_ner=addl_ner, rel_cats=rel_cats)
        logger.info(cat.get_model_card())  # Print the model card

        return cat

    @classmethod
    def load_cdb(cls, model_pack_path: str) -> CDB:
        """
        Loads the concept database from the provided model pack path

        Args:
            model_pack_path (str): path to model pack, zip or dir.

        Returns:
            CDB: The loaded concept database
        """
        cdb_path = os.path.join(model_pack_path, "cdb.dat")
        nr_of_jsons_expected = len(SPECIALITY_NAMES) - len(ONE2MANY)
        has_jsons = len(glob.glob(os.path.join(model_pack_path, '*.json'))) >= nr_of_jsons_expected
        json_path = model_pack_path if has_jsons else None
        logger.info('Loading model pack with %s', 'JSON format' if json_path else 'dill format')
        cdb = CDB.load(cdb_path, json_path)
        return cdb

    @classmethod
    def load_meta_cats(cls, model_pack_path: str, meta_cat_config_dict: Optional[Dict] = None) -> List[Tuple[str, MetaCAT]]:
        """

        Args:
            model_pack_path (str): path to model pack, zip or dir.
            meta_cat_config_dict (Optional[Dict]):
                A config dict that will overwrite existing configs in meta_cat.
                e.g. meta_cat_config_dict = {'general': {'device': 'cpu'}}.
                Defaults to None.

        Returns:
            List[Tuple(str, MetaCAT)]: list of pairs of meta cat model names (i.e. the task name) and the MetaCAT models.
        """
        meta_paths = [os.path.join(model_pack_path, path)
                      for path in os.listdir(model_pack_path) if path.startswith('meta_')]
        meta_cats = []
        for meta_path in meta_paths:
            meta_cats.append(MetaCAT.load(save_dir_path=meta_path,
                                          config_dict=meta_cat_config_dict))
        return list(zip(meta_paths, meta_cats))

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
        # Should we train - do not use this for training, unless you know what you are doing. Use the
        #self.train() function
        self.config.linking.train = do_train

        if text is None:
            logger.error("The input text should be either a string or a sequence of strings but got %s", type(text))
            return None
        else:
            text = str(text)  # NOTE: shouldn't be necessary but left it in
            if self.config.general.usage_monitor.enabled:
                l1 = len(text)
                text = self._get_trimmed_text(text)
                l2 = len(text)
                rval = self.pipe(text)
                # NOTE: pipe returns Doc (not List[Doc]) since we passed str (not List[str])
                #       that's why we ignore type here
                #       But it could still be None if the text is empty
                if rval is None:
                    nents = 0
                elif self.config.general.show_nested_entities:
                    nents = len(rval._.ents)  # type: ignore
                else:
                    nents = len(rval.ents)  # type: ignore
                self.usage_monitor.log_inference(l1, l2, nents)
                return rval  # type: ignore
            else:
                text = self._get_trimmed_text(text)
                return self.pipe(text)  # type: ignore

    def __repr__(self) -> str:
        """Prints the model_card for this CAT instance.

        Returns:
            str: the 'Model Card' for this CAT instance. This includes NER+L config and any MetaCATs
        """
        return self.get_model_card(as_dict=False)

    def _print_stats(self,
                     data: Dict,
                     epoch: int = 0,
                     use_project_filters: bool = False,
                     use_overlaps: bool = False,
                     use_cui_doc_limit: bool = False,
                     use_groups: bool = False,
                     extra_cui_filter: Optional[Set] = None,
                     do_print: bool = True) -> Tuple:
        """TODO: Refactor and make nice
        Print metrics on a dataset (F1, P, R), it will also print the concepts that have the most FP,FN,TP.

        Args:
            data (Dict):
                The json object that we get from MedCATtrainer on export.
            epoch (int):
                Used during training, so we know what epoch is it.
            use_project_filters (bool):
                Each project in MedCATtrainer can have filters, do we want to respect those filters
                when calculating metrics.
            use_overlaps (bool):
                Allow overlapping entities, nearly always False as it is very difficult to annotate overlapping entities.
            use_cui_doc_limit (bool):
                If True the metrics for a CUI will be only calculated if that CUI appears in a document, in other words
                if the document was annotated for that CUI. Useful in very specific situations when during the annotation
                process the set of CUIs changed.
            use_groups (bool):
                If True concepts that have groups will be combined and stats will be reported on groups.
            extra_cui_filter(Optional[Set]):
                This filter will be intersected with all other filters, or if all others are not set then only this one will be used.
            do_print (bool):
                Whether to print stats out. Defaults to True.

        Returns:
            fps (dict):
                False positives for each CUI.
            fns (dict):
                False negatives for each CUI.
            tps (dict):
                True positives for each CUI.
            cui_prec (dict):
                Precision for each CUI.
            cui_rec (dict):
                Recall for each CUI.
            cui_f1 (dict):
                F1 for each CUI.
            cui_counts (dict):
                Number of occurrence for each CUI.
            examples (dict):
                Examples for each of the fp, fn, tp. Format will be examples['fp']['cui'][<list_of_examples>].
        """
        return get_stats(self, data=data, epoch=epoch, use_project_filters=use_project_filters,
                         use_overlaps=use_overlaps, use_cui_doc_limit=use_cui_doc_limit,
                         use_groups=use_groups, extra_cui_filter=extra_cui_filter, do_print=do_print)

    def _init_ckpts(self, is_resumed, checkpoint):
        if self.config.general.checkpoint.steps is not None or checkpoint is not None:
            checkpoint_config = CheckpointConfig(**self.config.general.checkpoint.model_dump())
            checkpoint_manager = CheckpointManager('cat_train', checkpoint_config)
            if is_resumed:
                # TODO: probably remove is_resumed mark and always resume if a checkpoint is provided,
                #but I'll leave it for now
                checkpoint = checkpoint or checkpoint_manager.get_latest_checkpoint()
                logger.info(f"Resume training on the most recent checkpoint at {checkpoint.dir_path}...")
                self.cdb = checkpoint.restore_latest_cdb()
                self.cdb.config.merge_config(self.config.asdict())
                self.config = self.cdb.config
                self._create_pipeline(self.config)
            else:
                checkpoint = checkpoint or checkpoint_manager.create_checkpoint()
                logger.info(f"Start new training and checkpoints will be saved at {checkpoint.dir_path}...")

        return checkpoint

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
        if not fine_tune:
            logger.info("Removing old training data!")
            self.cdb.reset_training()
        checkpoint = self._init_ckpts(is_resumed, checkpoint)

        # cache train state
        _prev_train = self.config.linking.train

        latest_trained_step = checkpoint.count if checkpoint is not None else 0
        epochal_data_iterator = chain.from_iterable(repeat(data_iterator, nepochs))
        for line in islice(epochal_data_iterator, latest_trained_step, None):
            if line is not None and line:
                # Convert to string
                line = str(line).strip()

                try:
                    _ = self(line, do_train=True)
                except Exception as e:
                    logger.warning("LINE: '%s...' \t WAS SKIPPED", line[0:100])
                    logger.warning("BECAUSE OF: %s", str(e))
            else:
                logger.warning("EMPTY LINE WAS DETECTED AND SKIPPED")

            latest_trained_step += 1
            if latest_trained_step % progress_print == 0:
                logger.info("DONE: %s", str(latest_trained_step))
            if checkpoint is not None and checkpoint.steps is not None and latest_trained_step % checkpoint.steps == 0:
                checkpoint.save(cdb=self.cdb, count=latest_trained_step)

        self.config.linking.train = _prev_train

    def add_cui_to_group(self, cui: str, group_name: str) -> None:
        """Adds a CUI to a group, will appear in cdb.addl_info['cui2group']

        Args:
            cui (str):
                The concept to be added.
            group_name (str):
                The group to which the concept will be added.

        Examples:

            >>> cat.add_cui_to_group("S-17", 'pain')
        """

        # Add group_name
        self.cdb.addl_info['cui2group'][cui] = group_name

    def unlink_concept_name(self, cui: str, name: str, preprocessed_name: bool = False) -> None:
        """Unlink a concept name from the CUI (or all CUIs if full_unlink), removes the link from
        the Concept Database (CDB). As a consequence medcat will never again link the `name`
        to this CUI - meaning the name will not be detected as a concept in the future.

        Args:
            cui (str):
                The CUI from which the `name` will be removed.
            name (str):
                The span of text to be removed from the linking dictionary.
            preprocessed_name (bool):
                Whether the name being used is preprocessed.

        Examples:

            >>> # To never again link C0020538 to HTN
            >>> cat.unlink_concept_name('C0020538', 'htn', False)
        """

        cuis = [cui]
        if preprocessed_name:
            names = {name: {'nothing': 'nothing'}}
        else:
            names = prepare_name(name, self.pipe.spacy_nlp, {}, self.config)

        # If full unlink find all CUIs
        if self.config.general.full_unlink:
            logger.warning("In the config `full_unlink` is set to `True`. "
                           "Thus removing all CUIs linked to the specified name"
                           " (%s)", name)
            for n in names:
                cuis.extend(self.cdb.name2cuis.get(n, []))

        # Remove name from all CUIs
        for c in cuis:
            self.cdb._remove_names(cui=c, names=names.keys())

    def add_and_train_concept(self,
                              cui: str,
                              name: str,
                              spacy_doc: Optional[Doc] = None,
                              spacy_entity: Optional[Union[List[Token], Span]] = None,
                              ontologies: Set[str] = set(),
                              name_status: str = 'A',
                              type_ids: Set[str] = set(),
                              description: str = '',
                              full_build: bool = True,
                              negative: bool = False,
                              devalue_others: bool = False,
                              do_add_concept: bool = True) -> None:
        r"""Add a name to an existing concept, or add a new concept, or do not do anything if the name or concept already exists. Perform
        training if spacy_entity and spacy_doc are set.

        Args:
            cui (str):
                CUI of the concept.
            name (str):
                Name to be linked to the concept (in the case of MedCATtrainer this is simply the
                selected value in text, no preprocessing or anything needed).
            spacy_doc (spacy.tokens.Doc):
                Spacy representation of the document that was manually annotated.
            spacy_entity (Optional[Union[List[Token], Span]]):
                Given the spacy document, this is the annotated span of text - list of annotated tokens that are marked with this CUI.
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
            negative (bool):
                Is this a negative or positive example.
            devalue_others (bool):
                If set, cuis to which this name is assigned and are not `cui` will receive negative training given
                that negative=False.
            do_add_concept (bool):
                Whether to add concept to CDB.
        """
        names = prepare_name(name, self.pipe.spacy_nlp, {}, self.config)
        if not names and cui not in self.cdb.cui2preferred_name and name_status == 'P':
            logger.warning("No names were able to be prepared in CAT.add_and_train_concept "
                           "method. As such no preferred name will be able to be specifeid. "
                           "The CUI: '%s' and raw name: '%s'", cui, name)
        # Only if not negative, otherwise do not add the new name if in fact it should not be detected
        if do_add_concept and not negative:
            self.cdb._add_concept(cui=cui, names=names, ontologies=ontologies, name_status=name_status, type_ids=type_ids, description=description,
                                 full_build=full_build)

        if spacy_entity is not None and spacy_doc is not None:
            # Train Linking
            self.linker.context_model.train(cui=cui, entity=spacy_entity, doc=spacy_doc, negative=negative, names=names)  # type: ignore

            if not negative and devalue_others:
                # Find all cuis
                cuis = set()
                for n in names:
                    cuis.update(self.cdb.name2cuis.get(n, []))
                # Remove the cui for which we just added positive training
                if cui in cuis:
                    cuis.remove(cui)
                # Add negative training for all other CUIs that link to these names
                for _cui in cuis:
                    self.linker.context_model.train(cui=_cui, entity=spacy_entity, doc=spacy_doc, negative=True)  # type: ignore


    def train_supervised_from_json(self,
                                   data_path: str,
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
        """
        Run supervised training on a dataset from MedCATtrainer in JSON format.

        Refer to `train_supervised_raw` for more details.

        # noqa: DAR101
        # noqa: DAR201
        """
        with open(data_path) as f:
            data = json.load(f)
        return self.train_supervised_raw(data, reset_cui_count, nepochs,
                                         print_stats, use_filters, terminate_last,
                                         use_overlaps, use_cui_doc_limit, test_size,
                                         devalue_others, use_groups, never_terminate,
                                         train_from_false_positives, extra_cui_filter,
                                         retain_extra_cui_filter, checkpoint,
                                         retain_filters, is_resumed)

    def train_supervised_raw(self,
                             data: Dict[str, List[Dict[str, dict]]],
                             reset_cui_count: bool = False,
                             nepochs: int = 1,
                             print_stats: int = 0,
                             use_filters: bool = False,
                             terminate_last: bool = False,
                             use_overlaps: bool = False,
                             use_cui_doc_limit: bool = False,
                             test_size: float = 0,
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
            reset_cui_count (bool):
                Used for training with weight_decay (annealing). Each concept has a count that is there
                from the beginning of the CDB, that count is used for annealing. Resetting the count will
                significantly increase the training impact. This will reset the count only for concepts
                that exist in the the training data.
            nepochs (int):
                Number of epochs for which to run the training.
            print_stats (int):
                If > 0 it will print stats every print_stats epochs.
            use_filters (bool):
                Each project in medcattrainer can have filters, do we want to respect those filters
                when calculating metrics.
            terminate_last (bool):
                If true, concept termination will be done after all training.
            use_overlaps (bool):
                Allow overlapping entities, nearly always False as it is very difficult to annotate overlapping entities.
            use_cui_doc_limit (bool):
                If True the metrics for a CUI will be only calculated if that CUI appears in a document, in other words
                if the document was annotated for that CUI. Useful in very specific situations when during the annotation
                process the set of CUIs changed.
            test_size (float):
                If > 0 the data set will be split into train test based on this ration. Should be between 0 and 1.
                Usually 0.1 is fine.
            devalue_others(bool):
                Check add_name for more details.
            use_groups (bool):
                If True concepts that have groups will be combined and stats will be reported on groups.
            never_terminate (bool):
                If True no termination will be applied
            train_from_false_positives (bool):
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

        Raises:
            ValueError: If attempting to retain filters with while training over multiple projects.

        Returns:
            Tuple: Consisting of the following parts
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
        checkpoint = self._init_ckpts(is_resumed, checkpoint)

        # the config.linking.filters stuff is used directly in
        # medcat.linking.context_based_linker and medcat.linking.vector_context_model
        # as such, they need to be kept up to date with per-project filters
        # However, the original state needs to be kept track of
        # so that it can be restored after training
        orig_filters = self.config.linking.filters.copy_of()
        local_filters = self.config.linking.filters

        fp = fn = tp = p = r = f1 = examples = {}

        cui_counts = {}

        if retain_filters:
            # TODO - allow specifying number of project to retain?
            if len(data['projects']) > 1:
                raise ValueError('Cannot retain multiple (potentially) different filters from multiple projects')
            # will merge with local when loading in project

        if test_size == 0:
            logger.info("Running without a test set, or train==test")
            test_set = data
            train_set = data
        else:
            train_set, test_set, _, _ = make_mc_train_test(data, self.cdb, test_size=test_size)

        if print_stats > 0:
            fp, fn, tp, p, r, f1, cui_counts, examples = self._print_stats(test_set,
                                                                           use_project_filters=use_filters,
                                                                           use_cui_doc_limit=use_cui_doc_limit,
                                                                           use_overlaps=use_overlaps,
                                                                           use_groups=use_groups,
                                                                           extra_cui_filter=extra_cui_filter)
        if reset_cui_count:
            # Get all CUIs
            cuis = []
            for project in train_set['projects']:
                for doc in project['documents']:
                    doc_annotations = self._get_doc_annotations(doc)
                    for ann in doc_annotations:
                        cuis.append(ann['cui'])
            for cui in set(cuis):
                if cui in self.cdb.cui2count_train:
                    self.cdb.cui2count_train[cui] = 100

        # Remove entities that were terminated
        if not never_terminate:
            for project in train_set['projects']:
                for doc in project['documents']:
                    doc_annotations = self._get_doc_annotations(doc)
                    for ann in doc_annotations:
                        if ann.get('killed', False):
                            self.unlink_concept_name(ann['cui'], ann['value'])

        latest_trained_step = checkpoint.count if checkpoint is not None else 0
        current_epoch, current_project, current_document = self._get_training_start(train_set, latest_trained_step)

        for epoch in trange(current_epoch, nepochs, initial=current_epoch, total=nepochs, desc='Epoch', leave=False):
            # Print acc before training
            for idx_project in trange(current_project, len(train_set['projects']), initial=current_project, total=len(train_set['projects']), desc='Project', leave=False):
                project = train_set['projects'][idx_project]

                # if retain filters, but not the extra_cui_filters (and they exist),
                # then we need to do project filters alone, then retain, and only
                # then add the extra CUI filters
                if retain_filters and extra_cui_filter and not retain_extra_cui_filter:
                    # adding project filters without extra_cui_filters
                    set_project_filters(self.cdb.addl_info, local_filters, project, set(), use_filters)
                    orig_filters.merge_with(local_filters)
                    # adding extra_cui_filters, but NOT project filters
                    set_project_filters(self.cdb.addl_info, local_filters, project, extra_cui_filter, False)
                    # refrain from doing it again for subsequent epochs
                    retain_filters = False
                else:
                    # Set filters in case we are using the train_from_fp
                    set_project_filters(self.cdb.addl_info, local_filters, project, extra_cui_filter, use_filters)

                for idx_doc in trange(current_document, len(project['documents']), initial=current_document, total=len(project['documents']), desc='Document', leave=False):
                    doc = project['documents'][idx_doc]
                    spacy_doc: Doc = self(doc['text'])  # type: ignore

                    # Compatibility with old output where annotations are a list
                    doc_annotations = self._get_doc_annotations(doc)
                    for ann in doc_annotations:
                        if not ann.get('killed', False):
                            cui = ann['cui']
                            start = ann['start']
                            end = ann['end']
                            spacy_entity = tkns_from_doc(spacy_doc=spacy_doc, start=start, end=end)
                            deleted = ann.get('deleted', False)
                            if local_filters.check_filters(cui):
                                self.add_and_train_concept(cui=cui,
                                                        name=ann['value'],
                                                        spacy_doc=spacy_doc,
                                                        spacy_entity=spacy_entity,
                                                        negative=deleted,
                                                        devalue_others=devalue_others)
                    if train_from_false_positives:
                        fps: List[Span] = get_false_positives(doc, spacy_doc)

                        for fp in fps:  # type: ignore
                            fp_: Span = fp  # type: ignore
                            self.add_and_train_concept(cui=fp_._.cui,
                                                       name=fp_.text,
                                                       spacy_doc=spacy_doc,
                                                       spacy_entity=fp_,
                                                       negative=True,
                                                       do_add_concept=False)

                    latest_trained_step += 1
                    if checkpoint is not None and checkpoint.steps is not None and latest_trained_step % checkpoint.steps == 0:
                        checkpoint.save(self.cdb, latest_trained_step)
                # if retaining MCT filters AND (if they exist) extra_cui_filters
                if retain_filters:
                    orig_filters.merge_with(local_filters)
                    # refrain from doing it again for subsequent epochs
                    retain_filters = False

            if terminate_last and not never_terminate:
                # Remove entities that were terminated, but after all training is done
                for project in train_set['projects']:
                    for doc in project['documents']:
                        doc_annotations = self._get_doc_annotations(doc)
                        for ann in doc_annotations:
                            if ann.get('killed', False):
                                self.unlink_concept_name(ann['cui'], ann['value'])

            if print_stats > 0 and (epoch + 1) % print_stats == 0:
                fp, fn, tp, p, r, f1, cui_counts, examples = self._print_stats(test_set,
                                                                               epoch=epoch + 1,
                                                                               use_project_filters=use_filters,
                                                                               use_cui_doc_limit=use_cui_doc_limit,
                                                                               use_overlaps=use_overlaps,
                                                                               use_groups=use_groups,
                                                                               extra_cui_filter=extra_cui_filter)

        # reset the state of filters
        self.config.linking.filters = orig_filters

        return fp, fn, tp, p, r, f1, cui_counts, examples

    def get_entities(self,
                     text: str,
                     only_cui: bool = False,
                     addl_info: List[str] = ['cui2icd10', 'cui2ontologies', 'cui2snomed']) -> Dict:
        doc = self(text)
        out = self._doc_to_out(doc, only_cui, addl_info)  # type: ignore
        return out

    def get_entities_multi_texts(self,
                                 texts: Union[Iterable[str], Iterable[Tuple]],
                                 only_cui: bool = False,
                                 addl_info: List[str] = ['cui2icd10', 'cui2ontologies', 'cui2snomed'],
                                 n_process: Optional[int] = None,
                                 batch_size: Optional[int] = None) -> List[Dict]:
        """Get entities

        Args:
            texts (Union[Iterable[str], Iterable[Tuple]]): Text to be annotated
            only_cui (bool): Whether to only return CUIs. Defaults to False.
            addl_info (List[str]): Additional info. Defaults to ['cui2icd10', 'cui2ontologies', 'cui2snomed'].
            n_process (Optional[int]): Number of processes. Defaults to None.
            batch_size (Optional[int]): The size of a batch. Defaults to None.

        Raises:
            ValueError: If there's a known issue with multiprocessing.
            RuntimeError: If there's an unknown issue with multprocessing.

        Returns:
            List[Dict]: List of entity documents.
        """
        out: List[Dict] = []

        if n_process is None:
            texts_ = self._generate_trimmed_texts(texts)
            for text in texts_:
                out.append(self._doc_to_out(self(text), only_cui, addl_info))  # type: ignore
        else:
            self.pipe.set_error_handler(self._pipe_error_handler)
            try:
                texts_ = self._get_trimmed_texts(texts)
                if self.config.general.usage_monitor.enabled:
                    input_lengths: List[Tuple[int, int]] = []
                    for orig_text, trimmed_text in zip(texts, texts_):
                        if orig_text is None or trimmed_text is None:
                            l1, l2 = 0, 0
                        else:
                            l1 = len(orig_text)
                            l2 = len(trimmed_text)
                        input_lengths.append((l1, l2))
                docs = self.pipe.batch_multi_process(texts_, n_process, batch_size)

                for doc_nr, doc in tqdm(enumerate(docs), total=len(texts_)):
                    doc = None if doc.text.strip() == '' else doc
                    out.append(self._doc_to_out(doc, only_cui, addl_info, out_with_text=True))
                    if self.config.general.usage_monitor.enabled:
                        l1, l2 = input_lengths[doc_nr]
                        if doc is None:
                            nents = 0
                        elif self.config.general.show_nested_entities:
                            nents = len(doc._.ents)  # type: ignore
                        else:
                            nents = len(doc.ents)  # type: ignore
                        self.usage_monitor.log_inference(l1, l2, nents)

                # Currently spaCy cannot mark which pieces of texts failed within the pipe so be this workaround,
                # which also assumes texts are different from each others.
                if len(out) < len(texts_):
                    logger.warning("Found at least one failed batch and set output for enclosed texts to empty")
                    for i, text in enumerate(texts_):
                        if i == len(out):
                            out.append(self._doc_to_out(None, only_cui, addl_info))  # type: ignore
                        elif out[i].get('text', '') != text:
                            out.insert(i, self._doc_to_out(None, only_cui, addl_info))  # type: ignore

                cnf_annotation_output = getattr(self.config, 'annotation_output', {})
                if not (cnf_annotation_output.get('include_text_in_output', False)):
                    for o in out:
                        if o is not None:
                            o.pop('text', None)
            except RuntimeError as e:
                if e.args == ('_share_filename_: only available on CPU',):
                    raise ValueError("Issue while performing multiprocessing. "
                                     "This is mostly likely to happen when "
                                     "using NER models (i.e DeId). If that is "
                                     "the case you could either a) save the "
                                     "model on disk and then load it back up; "
                                     "or b) install cpu-only toch.") from e
                raise e
            finally:
                self.pipe.reset_error_handler()

        return out

    def get_json(self, text: str, only_cui: bool = False, addl_info: List[str]=['cui2icd10', 'cui2ontologies']) -> str:
        """Get output in json format

        Args:
            text (str): Text to be annotated
            only_cui (bool): Whether to only get CUIs. Defaults to False.
            addl_info (List[str]): Additional info. Defaults to ['cui2icd10', 'cui2ontologies'].

        Returns:
            str: Json with fields {'entities': <>, 'text': text}.
        """
        ents = self.get_entities(text, only_cui, addl_info=addl_info)['entities']
        out = {'annotations': ents, 'text': text}

        return json.dumps(out)

    @staticmethod
    def _get_training_start(train_set, latest_trained_step):
        total_steps_per_epoch = sum([1 for project in train_set['projects'] for _ in project['documents']])
        if total_steps_per_epoch == 0:
            raise ValueError("MedCATtrainer export contains no documents")
        current_epoch, last_step_in_epoch = divmod(latest_trained_step, total_steps_per_epoch)
        document_count = 0
        current_project = 0
        current_document = 0
        for idx_project, project in enumerate(train_set['projects']):
            for idx_doc, _ in enumerate(project['documents']):
                document_count += 1
                if document_count == last_step_in_epoch:
                    current_project = idx_project
                    current_document = idx_doc
                    break
            if current_project > 0:
                break
            current_document = 0
        return current_epoch, current_project, current_document

    def _separate_nn_components(self):
        # Loop though the models and check are there GPU devices
        nn_components = []
        for component in self.pipe.spacy_nlp.components:
            if isinstance(component[1], MetaCAT) or isinstance(component[1], TransformersNER):
                self.pipe.spacy_nlp.disable_pipe(component[0])
                nn_components.append(component)

        return nn_components

    def _run_nn_components(self, docs: Dict, nn_components: List, id2text: Dict) -> None:
        """This will add meta_anns in-place to the docs dict.

        # noqa: DAR101
        """
        logger.debug("Running GPU components separately")

        # First convert the docs into the fake spacy doc format
        spacy_docs = json_to_fake_spacy(docs, id2text=id2text)
        # Disable component locks also
        for name, component in nn_components:
            component.config.general['disable_component_lock'] = True

        # For meta_cat components
        for name, component in [c for c in nn_components if isinstance(c[1], MetaCAT)]:
            spacy_docs = component.pipe(spacy_docs)
        for spacy_doc in spacy_docs:
            for ent in spacy_doc.ents:
                docs[spacy_doc.id]['entities'][ent._.id]['meta_anns'].update(ent._.meta_anns)

    def _batch_generator(self, data: Iterable, batch_size_chars: int, skip_ids: Set = set()):
        docs = []
        char_count = 0
        for doc in data:
            if doc[0] not in skip_ids:
                char_count += len(str(doc[1]))
                docs.append(doc)
                if char_count < batch_size_chars:
                    continue
                yield docs
                docs = []
                char_count = 0

        if len(docs) > 0:
            yield docs

    def _save_docs_to_file(self, docs: Iterable, annotated_ids: List[str], save_dir_path: str, annotated_ids_path: Optional[str], part_counter: int = 0) -> int:
        path = os.path.join(save_dir_path, 'part_{}.pickle'.format(part_counter))
        pickle.dump(docs, open(path, "wb"))
        logger.info("Saved part: %s, to: %s", part_counter, path)
        part_counter = part_counter + 1 # Increase for save, as it should be what is the next part
        if annotated_ids_path is not None:
            pickle.dump((annotated_ids, part_counter), open(annotated_ids_path, 'wb'))
        return part_counter

    def multiprocessing_batch_char_size(self,
                                        data: Union[List[Tuple], Iterable[Tuple]],
                                        nproc: int = 2,
                                        batch_size_chars: int = 5000 * 1000,
                                        only_cui: bool = False,
                                        addl_info: List[str] = [],
                                        separate_nn_components: bool = True,
                                        out_split_size_chars: Optional[int] = None,
                                        save_dir_path: str = os.path.abspath(os.getcwd()),
                                        min_free_memory=0.1,
                                        min_free_memory_size: Optional[str] = None,
                                        enabled_progress_bar: bool = True) -> Dict:
        r"""Run multiprocessing for inference, if out_save_path and out_split_size_chars is used this will also continue annotating
        documents if something is saved in that directory.

        This method batches the data based on the number of characters as specified by user.

        PS: This method is unlikely to work on a Windows machine.

        Args:
            data:
                Iterator or array with format: [(id, text), (id, text), ...]
            nproc (int):
                Number of processors. Defaults to 8.
            batch_size_chars (int):
                Size of a batch in number of characters, this should be around: NPROC * average_document_length * 200.
                Defaults to 1000000.
            only_cui (bool):
                Whether to only return the CUIs rather than the full annotations. Dedfaults to False.
            addl_info (List[str]):
                The additional information. Defaults to [].
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
                If both `min_free_memory` and `min_free_memory_size` are set, a ValueError is raised.
                Defaults to 0.1.
            min_free_memory_size (Optional[str]):
                If set, the process will not start unless there's the specified amount of memory available.
                For reference, we would recommend at least 5GB of memory for a full SNOMED model. You can use
                human readable sizes (e.g 2GB, 2000MB and so on). If both `min_free_memory` and
                `min_free_memory_size` are set, a ValueError is raised. Defaults to None.
            enabled_progress_bar (bool):
                Whether to enabled the progress bar. Defaults to True.

        Raises:
            Exception: If multiprocessing cannot be done.
            ValueError: If both free memory specifiers are provided.

        Returns:
            Dict:
                {id: doc_json, id2: doc_json2, ...}, in case out_split_size_chars is used
                the last batch will be returned while that and all previous batches will be
                written to disk (out_save_dir).
        """
        for comp in self.pipe.spacy_nlp.components:
            if isinstance(comp[1], TransformersNER):
                raise Exception("Please do not use multiprocessing when running a transformer model for NER, run sequentially.")

        if min_free_memory_size is not None and min_free_memory != 0.1:
            raise ValueError("Unknown minimum memory size. "
                             f"Provided `min_free_memory`={min_free_memory} "
                             f"as well as `min_free_memory_size`={min_free_memory_size}. "
                             "Please only provide one of the two.")
        if min_free_memory_size:
            min_free_memory_size_mr = humanfriendly.parse_size(min_free_memory_size)
        else:
            min_free_memory_size_mr = None

        # Set max document length
        self.pipe.spacy_nlp.max_length = self.config.preprocessing.max_document_length

        if self._meta_cats and not separate_nn_components:
            # Hack for torch using multithreading, which is not good if not 
            #separate_nn_components, need for CPU runs only
            import torch
            torch.set_num_threads(1)

        nn_components = []
        if separate_nn_components:
            nn_components = self._separate_nn_components()

        if save_dir_path is not None:
            os.makedirs(save_dir_path, exist_ok=True)

        # "5" looks like a magic number here so better with comment about why the choice was made.
        internal_batch_size_chars = batch_size_chars // (5 * nproc)

        annotated_ids_path = os.path.join(save_dir_path, 'annotated_ids.pickle') if save_dir_path is not None else None
        if annotated_ids_path is not None and os.path.exists(annotated_ids_path):
            annotated_ids, part_counter = pickle.load(open(annotated_ids_path, 'rb'))
        else:
            annotated_ids = []
            part_counter = 0

        # for progress bar
        if hasattr(data, '__len__'):  # Check if data has length
            total_docs = len(data)  # type: ignore
            iterator = tqdm(data, desc="Processing", unit="batch", total=total_docs, disable=not enabled_progress_bar)
        else:
            total_docs = None
            iterator = tqdm(data, desc="Processing", unit="batch", disable=not enabled_progress_bar)

        docs = {}
        _start_time = time.time()
        _batch_counter = 0 # Used for splitting the output, counts batches between saves
        for batch in self._batch_generator(iterator, batch_size_chars, skip_ids=set(annotated_ids)):
            logger.info("Annotated until now: %s docs; Current BS: %s docs; Elapsed time: %.2f minutes",
                          len(annotated_ids),
                          len(batch),
                          (time.time() - _start_time)/60)
            try:
                _docs = self._multiprocessing_batch(data=batch,
                                                    nproc=nproc,
                                                    only_cui=only_cui,
                                                    batch_size_chars=internal_batch_size_chars,
                                                    addl_info=addl_info,
                                                    nn_components=nn_components,
                                                    min_free_memory=min_free_memory,
                                                    min_free_memory_size=min_free_memory_size_mr)
                docs.update(_docs)
                annotated_ids.extend(_docs.keys())
                _batch_counter += 1
                del _docs
                if out_split_size_chars is not None and (_batch_counter * batch_size_chars) > out_split_size_chars:
                    # Save to file and reset the docs 
                    part_counter = self._save_docs_to_file(docs=docs,
                                                           annotated_ids=annotated_ids,
                                                           save_dir_path=save_dir_path,
                                                           annotated_ids_path=annotated_ids_path,
                                                           part_counter=part_counter)
                    del docs
                    docs = {}
                    _batch_counter = 0
                if total_docs is not None:
                    iterator.set_postfix({"Processed": len(annotated_ids), "Total": total_docs})
            except Exception as e:
                logger.warning("Failed an outer batch in the multiprocessing script")
                logger.warning(e, exc_info=True, stack_info=True)

        # Save the last batch
        if out_split_size_chars is not None and len(docs) > 0:
            # Save to file and reset the docs 
            self._save_docs_to_file(docs=docs,
                                    annotated_ids=annotated_ids,
                                    save_dir_path=save_dir_path,
                                    annotated_ids_path=annotated_ids_path,
                                    part_counter=part_counter)

        # Enable the GPU Components again
        if separate_nn_components:
            for name, _ in nn_components:
                # No need to do anything else as it was already in the pipe
                self.pipe.spacy_nlp.enable_pipe(name)

        return docs

    def _multiprocessing_batch(self,
                               data: Union[List[Tuple], Iterable[Tuple]],
                               nproc: int = 8,
                               batch_size_chars: int = 1000000,
                               only_cui: bool = False,
                               addl_info: List[str] = [],
                               nn_components: List = [],
                               min_free_memory: float = 0.1,
                               min_free_memory_size: Optional[int] = None) -> Dict:
        """Run multiprocessing on one batch.

        Args:
            data:
                Iterator or array with format: [(id, text), (id, text), ...].
            nproc (int):
                Number of processors. Defaults to 8.
            batch_size_chars (int):
                Size of a batch in number of characters. Fefaults to 1 000 000.
            only_cui (bool):
                Whether to get only CUIs. Defaults to False.
            addl_info (List[str]):
                Additional info. Defaults to [].
            nn_components (List):
                NN components in case there's a separation. Defaults to [].
            min_free_memory (float):
                If set a process will not start unless there is at least this much RAM memory left,
                should be a range between [0, 1] meaning how much of the memory has to be free. Helps when annotating
                very large datasets because spacy is not the best with memory management and multiprocessing.
                Defaults to 0.
            min_free_memory_size (Optional[int]):
                The minimum human readable memory size required.

        Returns:
            Dict:
                {id: doc_json, id2: doc_json2, ...}
        """
        # Create the input output for MP
        with Manager() as manager:
            out_list = manager.list()
            lock = manager.Lock()
            in_q = manager.Queue(maxsize=10*nproc)

            id2text = {}
            for batch in self._batch_generator(data, batch_size_chars):
                if nn_components:
                    # We need this for the json_to_fake_spacy
                    id2text.update({k: v for k, v in batch})
                in_q.put(batch)

            # Final data point for workers
            for _ in range(nproc):
                in_q.put(None)
            sleep(2)

            # Create processes
            procs = []
            for i in range(nproc):
                p = Process(target=self._mp_cons,
                            kwargs={'in_q': in_q,
                                    'out_list': out_list,
                                    'pid': i,
                                    'only_cui': only_cui,
                                    'addl_info': addl_info,
                                    'min_free_memory': min_free_memory,
                                    'min_free_memory_size': min_free_memory_size,
                                    'lock': lock})
                p.start()
                procs.append(p)

            # Join processes
            for p in procs:
                p.join()

            docs = {}
            # Converts a tuple into a dict
            docs.update({k: v for k, v in out_list})

        # If we have separate GPU components now we pipe that
        if nn_components:
            try:
                self._run_nn_components(docs, nn_components, id2text=id2text)
            except Exception as e:
                logger.warning(e, exc_info=True, stack_info=True)

        return docs

    def multiprocessing_batch_docs_size(self,
                                        in_data: Union[List[Tuple], Iterable[Tuple]],
                                        nproc: Optional[int] = None,
                                        batch_size: Optional[int] = None,
                                        only_cui: bool = False,
                                        addl_info: List[str] = ['cui2icd10', 'cui2ontologies', 'cui2snomed'],
                                        return_dict: bool = True,
                                        batch_factor: int = 2) -> Union[List[Tuple], Dict]:
        """Run multiprocessing NOT FOR TRAINING.

        This method batches the data based on the number of documents as specified by the user.

        NOTE: When providing a generator for `data`, the generator is evaluated (`list(in_data)`)
              and thus all the data is kept in memory and (potentially) duplicated for use in
              multiple threads. So if you're using a lot of data, it may be better to use
              `CAT.multiprocessing_batch_char_size` instead.

        PS:
        This method supports Windows.

        Args:
            in_data (Union[List[Tuple], Iterable[Tuple]]): List with format: [(id, text), (id, text), ...]
            nproc (Optional[int]): The number of processors. Defaults to None.
            batch_size (Optional[int]): The number of texts to buffer. Defaults to None.
            only_cui (bool): Whether to get only CUIs. Defaults to False.
            addl_info (List[str]): Additional info. Defaults to [].
            return_dict (bool): Flag for returning either a dict or a list of tuples. Defaults to True.
            batch_factor (int): Batch factor. Defaults to 2.

        Raises:
            ValueError: When number of processes is 0.

        Returns:
            Union[List[Tuple], Dict]:
                {id: doc_json, id: doc_json, ...} or if return_dict is False, a list of tuples: [(id, doc_json), (id, doc_json), ...]
        """
        out: Union[Dict, List[Tuple]]

        if nproc == 0:
            raise ValueError("nproc cannot be set to zero")

        # TODO: Surely there's a way to not materialise all of the incoming data in memory?
        #       This is counter productive for allowing the passing of generators.
        if isinstance(in_data, Iterable):
            in_data = list(in_data)
            in_data_len = len(in_data)
            if in_data_len > MIN_GEN_LEN_FOR_WARN:
                # only point this out when it's relevant, i.e over 10k items
                logger.warning("The `CAT.multiprocessing_batch_docs_size` method just "
                               f"materialised {in_data_len} items from the generator it "
                               "was provided. This may use up a considerable amount of "
                               "RAM, especially since the data may be duplicated across "
                               "multiple threads when multiprocessing is used. If the "
                               "process is kiled after this warning, please use the "
                               "alternative method `multiprocessing_batch_char_size` instead")
        n_process = nproc if nproc is not None else min(max(cpu_count() - 1, 1), math.ceil(len(in_data) / batch_factor))
        batch_size = batch_size if batch_size is not None else math.ceil(len(in_data) / (batch_factor * abs(n_process)))

        start_method = None
        try:
            if self._meta_cats:
                import torch
                if torch.multiprocessing.get_start_method() != "spawn":
                    start_method = torch.multiprocessing.get_start_method()
                    torch.multiprocessing.set_start_method("spawn", force=True)

            entities = self.get_entities_multi_texts(texts=in_data, only_cui=only_cui, addl_info=addl_info,
                                                     n_process=n_process, batch_size=batch_size)
        finally:
            if start_method is not None:
                import torch
                torch.multiprocessing.set_start_method(start_method, force=True)

        if return_dict:
            out = {}
            for idx, data in enumerate(in_data):
                out[data[0]] = entities[idx]
        else:
            out = []
            for idx, data in enumerate(in_data):
                out.append((data[0], entities[idx]))

        return out

    def _mp_cons(self, in_q: Queue, out_list: List, min_free_memory: float,
                 lock: Lock, min_free_memory_size: Optional[int] = None,
                 pid: int = 0, only_cui: bool = False, addl_info: List = []) -> None:
        if min_free_memory_size is not None:
            # passed as int not str
            min_free_memory_mr = min_free_memory_size
        else:
            min_free_memory_mr = min_free_memory * psutil.virtual_memory().total
        out: List = []

        while True:
            if not in_q.empty():
                if psutil.virtual_memory().available < min_free_memory_mr:
                    with lock:
                        out_list.extend(out)
                    # Stop a process if there is not enough memory left
                    virmem = psutil.virtual_memory()
                    logger.warning("Stopping multiprocessing because there is no enough memory available. "
                                   "Currently %s of memory (out of %s) memory (a fraction of %3.2f) "
                                   "is available but a minimum of %s is required "
                                   "(from %3.2f fraction or %s specified size). "
                                   "If you believe you have enough memory, you can change the `min_free_memory` "
                                   "or `min_free_memory_size` with latter preferred (but not both!) "
                                   "keyword argument to something lower. For reference, We would recommend a "
                                   "minimum of 5GB of memory for a full SNOMED model.",
                                   humanfriendly.format_size(virmem.available), humanfriendly.format_size(virmem.total),
                                   virmem.available / virmem.total, humanfriendly.format_size(min_free_memory_mr),
                                   min_free_memory, str(min_free_memory_size))
                    break

                data = in_q.get()
                if data is None:
                    with lock:
                        out_list.extend(out)
                    break

                for i_text, text in data:
                    try:
                        # Annotate document
                        doc = self.get_entities(text=text, only_cui=only_cui, addl_info=addl_info)
                        out.append((i_text, doc))
                    except Exception as e:
                        logger.warning("PID: %s failed one document in _mp_cons, running will continue normally. \n" +
                                         "Document length in chars: %s, and ID: %s", pid, len(str(text)), i_text)
                        logger.warning(str(e))
        if self.config.general.usage_monitor.enabled:
            # NOTE: This is in another process, so need to explicitly flush
            self.usage_monitor._flush_logs()
        sleep(2)

    def _add_nested_ent(self, doc: Doc, _ents: List[Span], _ent: Union[Dict, Span]) -> None:
        # if the entities are serialised (PipeRunner.serialize_entities)
        # then the entities are dicts
        # otherwise they're Span objects
        meta_anns = None
        if isinstance(_ent, dict):
            start = _ent['start']
            end =_ent['end']
            label = _ent['label']
            cui = _ent['cui']
            detected_name = _ent['detected_name']
            context_similarity = _ent['context_similarity']
            id = _ent['id']
            if 'meta_anns' in _ent:
                meta_anns = _ent['meta_anns']
        else:
            start = _ent.start
            end = _ent.end
            label = _ent.label
            cui = _ent._.cui
            detected_name = _ent._.detected_name
            context_similarity = _ent._.context_similarity
            if _ent._.has('meta_anns'):
                meta_anns = _ent._.meta_anns
            if HAS_NEW_SPACY:
                id = _ent.id
            else:
                id = _ent.ent_id
        entity = Span(doc, start, end, label=label)
        entity._.cui = cui
        entity._.detected_name = detected_name
        entity._.context_similarity = context_similarity
        entity._.id = id
        if meta_anns is not None:
            entity._.meta_anns = meta_anns
        _ents.append(entity)

    def _doc_to_out(self,
                    doc: Doc,
                    only_cui: bool,
                    addl_info: List[str],
                    out_with_text: bool = False) -> Dict:
        out: Dict = {'entities': {}, 'tokens': []}
        cnf_annotation_output = self.config.annotation_output
        if doc is not None:
            out_ent: Dict = {}
            if self.config.general.show_nested_entities:
                _ents: List[Span] = []
                for _ent in doc._.ents:
                    self._add_nested_ent(doc, _ents, _ent)
            else:
                _ents = doc.ents  # type: ignore

            if cnf_annotation_output.lowercase_context:
                doc_tokens = [tkn.text_with_ws.lower() for tkn in list(doc)]
            else:
                doc_tokens = [tkn.text_with_ws for tkn in list(doc)]

            if cnf_annotation_output.doc_extended_info:
                # Add tokens if extended info
                out['tokens'] = doc_tokens

            context_left = cnf_annotation_output.context_left
            context_right = cnf_annotation_output.context_right
            doc_extended_info = cnf_annotation_output.doc_extended_info

            for _, ent in enumerate(_ents):
                cui = str(ent._.cui)
                if not only_cui:
                    out_ent['pretty_name'] = self.cdb.get_name(cui)
                    out_ent['cui'] = cui
                    out_ent['type_ids'] = list(self.cdb.cui2type_ids.get(cui, ''))
                    out_ent['types'] = [self.cdb.addl_info['type_id2name'].get(tui, '') for tui in out_ent['type_ids']]
                    out_ent['source_value'] = ent.text
                    out_ent['detected_name'] = str(ent._.detected_name)
                    out_ent['acc'] = float(ent._.context_similarity)
                    out_ent['context_similarity'] = float(ent._.context_similarity)
                    out_ent['start'] = ent.start_char
                    out_ent['end'] = ent.end_char
                    for addl in addl_info:
                        tmp = self.cdb.addl_info.get(addl, {}).get(cui, [])
                        out_ent[addl.split("2")[-1]] = list(tmp) if type(tmp) is set else tmp
                    out_ent['id'] = ent._.id
                    out_ent['meta_anns'] = {}

                    if doc_extended_info:
                        out_ent['start_tkn'] = ent.start
                        out_ent['end_tkn'] = ent.end

                    if context_left > 0 and context_right > 0:
                        out_ent['context_left'] = doc_tokens[max(ent.start - context_left, 0):ent.start]
                        out_ent['context_right'] = doc_tokens[ent.end:min(ent.end + context_right, len(doc_tokens))]
                        out_ent['context_center'] = doc_tokens[ent.start:ent.end]

                    if hasattr(ent._, 'meta_anns') and ent._.meta_anns:
                        out_ent['meta_anns'] = ent._.meta_anns

                    out['entities'][out_ent['id']] = dict(out_ent)
                else:
                    out['entities'][ent._.id] = cui

            if cnf_annotation_output.include_text_in_output or out_with_text:
                out['text'] = doc.text
        return out

    def _get_trimmed_text(self, text: Optional[str]) -> str:
        return text[0:self.config.preprocessing.max_document_length] if text is not None and len(text) > 0 else ""

    def _generate_trimmed_texts(self, texts: Union[Iterable[str], Iterable[Tuple]]) -> Iterable[str]:
        text_: str
        for text in texts:
            text_ = text[1] if isinstance(text, tuple) else text
            yield self._get_trimmed_text(text_)

    def _get_trimmed_texts(self, texts: Union[Iterable[str], Iterable[Tuple]]) -> List[str]:
        trimmed: List = []
        text_: str
        for text in texts:
            text_ = text[1] if isinstance(text, tuple) else text
            trimmed.append(self._get_trimmed_text(text_))
        return trimmed

    @staticmethod
    def _pipe_error_handler(proc_name: str, proc: "Pipe", docs: List[Doc], e: Exception) -> None:
        logger.warning("Exception raised when applying component %s to a batch of docs.", proc_name)
        logger.warning(e, exc_info=True, stack_info=True)
        if docs is not None:
            logger.warning("Docs contained in the batch:")
            for doc in docs:
                if hasattr(doc, "text"):
                    logger.warning("%s...", doc.text[:50])

    @staticmethod
    def _get_doc_annotations(doc: Doc):
        if type(doc['annotations']) is list:  # type: ignore
            return doc['annotations']  # type: ignore
        if type(doc['annotations']) is dict:  # type: ignore
            return doc['annotations'].values()  # type: ignore
        return None

    def destroy_pipe(self):
        self.pipe.destroy()
