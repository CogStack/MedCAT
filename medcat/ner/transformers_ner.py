import os
import json
import logging
from spacy.tokens import Doc
from datetime import datetime
from typing import Iterable, Iterator, Optional, Dict, List, cast, Union
from spacy.tokens import Span

from medcat.cdb import CDB
from medcat.utils.meta_cat.ml_utils import set_all_seeds
from medcat.datasets import transformers_ner
from medcat.utils.postprocessing import map_ents_to_groups, make_pretty_labels, create_main_ann, LabelStyle
from medcat.utils.hasher import Hasher
from medcat.config_transformers_ner import ConfigTransformersNER
from medcat.tokenizers.transformers_ner import TransformersTokenizerNER
from medcat.utils.ner.metrics import metrics
from medcat.datasets.data_collator import CollateAndPadNER

from transformers import Trainer, AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline, TrainingArguments
import datasets

# It should be safe to do this always, as all other multiprocessing
#will be finished before data comes to meta_cat
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['WANDB_DISABLED'] = 'true'


logger = logging.getLogger(__name__)


class TransformersNER(object):
    """TODO: Add documentation"""

    # Custom pipeline component name
    name = 'transformers_ner'

    def __init__(self, cdb, config: Optional[ConfigTransformersNER] = None,
                 training_arguments=None) -> None:
        self.cdb = cdb
        if config is None:
            config = ConfigTransformersNER()

        self.config = config
        set_all_seeds(config.general['seed'])

        self.model = AutoModelForTokenClassification.from_pretrained(config.general['model_name'])

        # Get the tokenizer either create a new one or load existing
        if os.path.exists(os.path.join(config.general['model_name'], 'tokenizer.dat')):
            self.tokenizer = TransformersTokenizerNER.load(os.path.join(config.general['model_name'], 'tokenizer.dat'))
        else:
            hf_tokenizer = AutoTokenizer.from_pretrained(self.config.general['model_name'])
            self.tokenizer = TransformersTokenizerNER(hf_tokenizer)

        if training_arguments is None:
            self.training_arguments = TrainingArguments(
                output_dir='./results',
                logging_dir='./logs',            # directory for storing logs
                num_train_epochs=10,              # total number of training epochs
                per_device_train_batch_size=1,  # batch size per device during training
                per_device_eval_batch_size=1,   # batch size for evaluation
                weight_decay=0.14,               # strength of weight decay
                warmup_ratio=0.01,
                learning_rate=4.47e-05, # Should be smaller when finetuning an existing deid model
                eval_accumulation_steps=1,
                gradient_accumulation_steps=4, # We want to get to bs=4
                do_eval=True,
                evaluation_strategy='epoch', # type: ignore
                logging_strategy='epoch', # type: ignore
                save_strategy='epoch', # type: ignore
                metric_for_best_model='eval_recall', # Can be changed if our preference is not recall but precision or f1
                load_best_model_at_end=True,
                remove_unused_columns=False)
        else:
            self.training_arguments = training_arguments


    def create_eval_pipeline(self):
        self.ner_pipe = pipeline(model=self.model, task="ner", tokenizer=self.tokenizer.hf_tokenizer)
        self.ner_pipe.device = self.model.device

    def get_hash(self):
        """A partial hash trying to catch differences between models."""
        hasher = Hasher()
        # Set last_train_on if None
        if self.config.general['last_train_on'] is None:
            self.config.general['last_train_on'] = datetime.now().timestamp()

        hasher.update(self.config.get_hash())
        return hasher.hexdigest()

    def _prepare_dataset(self, json_path, ignore_extra_labels, meta_requirements, file_name='data.json'):
        def merge_data_loaded(base, other):
            if not base:
                return other
            elif other is None:
                return base
            else:
                for p in other['projects']:
                    base['projects'].append(p)
            return base

        if isinstance(json_path, str):
            json_path = [json_path]

        # Merge data from all different data paths
        data_loaded: Dict = {}
        for path in json_path:
            with open(path, 'r') as f:
                data_loaded = merge_data_loaded(data_loaded, json.load(f))

        # Remove labels that did not exist in old dataset
        if ignore_extra_labels and self.tokenizer.label_map:
            logger.info("Ignoring extra labels from the data")
            for p in data_loaded['projects']:
                for d in p['documents']:
                    new_anns = []
                    for a in d['annotations']:
                        if a['cui'] in self.tokenizer.label_map:
                            new_anns.append(a)
                    d['annotations'] = new_anns
        if meta_requirements is not None:
            logger.info("Removing anns that do not meet meta requirements")
            for p in data_loaded['projects']:
                for d in p['documents']:
                    new_anns = []
                    for a in d['annotations']:
                        if all([a['meta_anns'][name]['value'] == value for name, value in meta_requirements.items()]):
                            new_anns.append(a)
                    d['annotations'] = new_anns

        # Here we have to save the data because of the data loader
        os.makedirs('results', exist_ok=True)
        out_path = os.path.join(os.getcwd(), 'results', file_name)
        json.dump(data_loaded, open(out_path, 'w'))

        return out_path

    def train(self, json_path: Union[str, list, None]=None, ignore_extra_labels=False, dataset=None, meta_requirements=None):
        """Train or continue training a model give a json_path containing a MedCATtrainer export. It will
        continue training if an existing model is loaded or start new training if the model is blank/new.

        Args:
            json_path (str or list):
                Path/Paths to a MedCATtrainer export containing the meta_annotations we want to train for.
            train_arguments(str):
                HF TrainingArguments. If None the default will be used. Defaults to `None`.
            ignore_extra_labels:
                Makes only sense when an existing deid model was loaded and from the new data we want to ignore
                labels that did not exist in the old model.
        """

        if dataset is None and json_path is not None:
            # Load the medcattrainer export
            json_path = self._prepare_dataset(json_path, ignore_extra_labels=ignore_extra_labels,
                                              meta_requirements=meta_requirements, file_name='data_eval.json')
            # Load dataset
            dataset = datasets.load_dataset(os.path.abspath(transformers_ner.__file__),
                                            data_files={'train': json_path}, # type: ignore
                                            split='train',
                                            cache_dir='/tmp/')
            # We split before encoding so the split is document level, as encoding
            #does the document spliting into max_seq_len
            dataset = dataset.train_test_split(test_size=self.config.general['test_size']) # type: ignore

        # Update labelmap in case the current dataset has more labels than what we had before
        self.tokenizer.calculate_label_map(dataset['train'])
        self.tokenizer.calculate_label_map(dataset['test'])

        if self.model.num_labels != len(self.tokenizer.label_map):
            logger.warning("The dataset contains labels we've not seen before, model is being reinitialized")
            logger.warning("Model: {} vs Dataset: {}".format(self.model.num_labels, len(self.tokenizer.label_map)))
            self.model = AutoModelForTokenClassification.from_pretrained(self.config.general['model_name'], num_labels=len(self.tokenizer.label_map))
            self.tokenizer.cui2name = {k:self.cdb.get_name(k) for k in self.tokenizer.label_map.keys()}

        self.model.config.id2label = {v:k for k,v in self.tokenizer.label_map.items()}
        self.model.config.label2id = self.tokenizer.label_map


        # Encode dataset
        encoded_dataset = dataset.map(
                lambda examples: self.tokenizer.encode(examples, ignore_subwords=False),
                batched=True,
                remove_columns=['ent_cuis', 'ent_ends', 'ent_starts', 'text'])

        data_collator = CollateAndPadNER(self.tokenizer.hf_tokenizer.pad_token_id) # type: ignore
        trainer = Trainer(
                model=self.model,
                args=self.training_arguments,
                train_dataset=encoded_dataset['train'],
                eval_dataset=encoded_dataset['test'],
                compute_metrics=lambda p: metrics(p, tokenizer=self.tokenizer, dataset=encoded_dataset['test'], verbose=self.config.general['verbose_metrics']),
                data_collator=data_collator, # type: ignore
                tokenizer=None)

        trainer.train() # type: ignore

        # Save the training time
        self.config.general['last_train_on'] = datetime.now().timestamp() # type: ignore

        # Save everything
        self.save(save_dir_path=os.path.join(self.training_arguments.output_dir, 'final_model'))

        # Run an eval step and return metrics
        p = trainer.predict(encoded_dataset['test']) # type: ignore
        df, examples = metrics(p, return_df=True, tokenizer=self.tokenizer, dataset=encoded_dataset['test'])


        # Create the pipeline for eval
        self.create_eval_pipeline()

        return df, examples, dataset

    def eval(self, json_path: Union[str, list, None] = None, dataset=None, ignore_extra_labels=False, meta_requirements=None):
        if dataset is None:
            json_path = self._prepare_dataset(json_path, ignore_extra_labels=ignore_extra_labels,
                                              meta_requirements=meta_requirements, file_name='data_eval.json')
            # Load dataset
            dataset = datasets.load_dataset(os.path.abspath(transformers_ner.__file__),
                                            data_files={'train': json_path}, # type: ignore
                                            split='train',
                                            cache_dir='/tmp/')

        # Encode dataset
        encoded_dataset = dataset.map(
                lambda examples: self.tokenizer.encode(examples, ignore_subwords=False),
                batched=True,
                remove_columns=['ent_cuis', 'ent_ends', 'ent_starts', 'text'])

        data_collator = CollateAndPadNER(self.tokenizer.hf_tokenizer.pad_token_id) # type: ignore
        # TODO: switch from trainer to model prediction
        trainer = Trainer(
                model=self.model,
                args=self.training_arguments,
                train_dataset=None,
                eval_dataset=encoded_dataset, # type: ignore
                compute_metrics=None,
                data_collator=data_collator, # type: ignore
                tokenizer=None)

        # Run an eval step and return metrics
        p = trainer.predict(encoded_dataset) # type: ignore
        df, examples = metrics(p, return_df=True, tokenizer=self.tokenizer, dataset=encoded_dataset)

        return df, examples

    def save(self, save_dir_path: str) -> None:
        """Save all components of this class to a file

        Args:
            save_dir_path(str):
                Path to the directory where everything will be saved.
        """
        # Create dirs if they do not exist
        os.makedirs(save_dir_path, exist_ok=True)

        # Save tokenizer
        self.tokenizer.save(os.path.join(save_dir_path, 'tokenizer.dat'))

        # Save config
        self.config.save(os.path.join(save_dir_path, 'cat_config.json'))

        # Save the model
        self.model.save_pretrained(save_dir_path)

        # Save the cdb
        self.cdb.save(os.path.join(save_dir_path, 'cdb.dat'))

        # This is everything we need to save from the class, we do not
        #save the class itself.

    @classmethod
    def load(cls, save_dir_path: str, config_dict: Optional[Dict] = None) -> "TransformersNER":
        """Load a meta_cat object.

        Args:
            save_dir_path (str):
                The directory where all was saved.
            config_dict (dict):
                This can be used to overwrite saved parameters for this meta_cat
                instance. Why? It is needed in certain cases where we autodeploy stuff.

        Returns:
            meta_cat (`medcat.MetaCAT`):
                You don't say
        """

        # Load config
        config = cast(ConfigTransformersNER, ConfigTransformersNER.load(os.path.join(save_dir_path, 'cat_config.json')))
        config.general['model_name'] = save_dir_path

        # Overwrite loaded paramters with something new
        if config_dict is not None:
            config.merge_config(config_dict)

        # Load cdb
        cdb = CDB.load(os.path.join(save_dir_path, 'cdb.dat'))

        ner = cls(cdb=cdb, config=config)
        ner.create_eval_pipeline()

        return ner

    @staticmethod
    def batch_generator(stream: Iterable[Doc], batch_size_chars: int) -> Iterable[List[Doc]]:
        docs = []
        char_count = 0
        for doc in stream:
            char_count += len(doc.text)
            docs.append(doc)
            if char_count < batch_size_chars:
                continue
            yield docs
            docs = []
            char_count = 0

        # If there is anything left return that also
        if len(docs) > 0:
            yield docs

    def pipe(self, stream: Iterable[Union[Doc, None]], *args, **kwargs) -> Iterator[Doc]:
        """Process many documents at once.

        Args:
            stream (Iterable[spacy.tokens.Doc]):
                List of spacy documents.
        """
        # Just in case
        if stream is None or not stream:
            return stream

        batch_size_chars = self.config.general['pipe_batch_size_in_chars']
        yield from self._process(stream, batch_size_chars)  # type: ignore

    def _process(self,
                 stream: Iterable[Union[Doc, None]],
                 batch_size_chars: int) -> Iterator[Optional[Doc]]:
        for docs in self.batch_generator(stream, batch_size_chars):  # type: ignore
            #all_text = [doc.text for doc in docs]
            #all_text_processed = self.tokenizer.encode_eval(all_text)
            # For now we will process the documents one by one, should be improved in the future to use batching
            for doc in docs:
                try:
                    res = self.ner_pipe(doc.text, aggregation_strategy=self.config.general['ner_aggregation_strategy'])
                    doc.ents = []  # type: ignore
                    for r in res:
                        inds = []
                        for ind, word in enumerate(doc):
                            end_char = word.idx + len(word.text)
                            if end_char <= r['end'] and end_char > r['start']:
                                inds.append(ind)
                            # To not loop through everything
                            if end_char > r['end']:
                                break
                        if inds:
                            entity = Span(doc, min(inds), max(inds) + 1, label=r['entity_group'])
                            entity._.cui = r['entity_group']
                            entity._.context_similarity = r['score']
                            entity._.detected_name = r['word']
                            entity._.id = len(doc._.ents)
                            entity._.confidence = r['score']

                            doc._.ents.append(entity)
                    create_main_ann(self.cdb, doc)
                    if self.cdb.config.general['make_pretty_labels'] is not None:
                        make_pretty_labels(self.cdb, doc, LabelStyle[self.cdb.config.general['make_pretty_labels']])
                    if self.cdb.config.general['map_cui_to_group'] is not None and self.cdb.addl_info.get('cui2group', {}):
                        map_ents_to_groups(self.cdb, doc)
                except Exception as e:
                    logger.warning(e, exc_info=True)
            yield from docs

    # Override
    def __call__(self, doc: Doc) -> Doc:
        """Process one document, used in the spacy pipeline for sequential
        document processing.

        Args:
            doc (spacy.tokens.Doc):
                A spacy document
        """

        # Just call the pipe method
        doc = next(self.pipe(iter([doc])))

        return doc
