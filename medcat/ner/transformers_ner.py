import os
import json
import logging
import torch
import numpy
from multiprocessing import Lock
from torch import nn, Tensor
from spacy.tokens import Doc
from datetime import datetime
from typing import Iterable, Iterator, Optional, Dict, List, Tuple, cast, Union
from medcat.utils.hasher import Hasher
from medcat.config_transformers_ner import ConfigTransformersNER
from medcat.utils.meta_cat.ml_utils import predict, train_model, set_all_seeds, eval_model
from medcat.utils.meta_cat.data_utils import prepare_from_json, encode_category_values
from medcat.utils.loggers import add_handlers
from medcat.tokenizers.transformers_ner import TransformersTokenizerNER
from medcat.utils.deid.metrics import metrics
from medcat.datasets.data_collator import CollateAndPadNER
from transformers import pipeline
from spacy.tokens import Span
import datasets

from medcat.cdb import CDB
from medcat.datasets import transformers_ner
from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, AutoTokenizer

# It should be safe to do this always, as all other multiprocessing
#will be finished before data comes to meta_cat
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['WANDB_DISABLED'] = 'true'


class TransformersNER(object):
    r''' TODO: Add documentation
    '''

    # Custom pipeline component name
    name = 'transformers_ner'

    # Add file and console handlers
    log = add_handlers(logging.getLogger(__package__))

    def __init__(self, cdb, config: Optional[ConfigTransformersNER] = None) -> None:
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

    def create_eval_pipeline(self):
        self.ner_pipe = pipeline(model=self.model, task="ner", tokenizer=self.tokenizer.hf_tokenizer, config=self.config.train)
        self.ner_pipe.device = self.model.device

    def get_hash(self):
        r''' A partial hash trying to catch differences between models
        '''
        hasher = Hasher()
        # Set last_train_on if None
        if self.config.train.last_train_on is None:
            self.config.train.last_train_on = datetime.now().timestamp()

        hasher.update(self.config.get_hash())
        return hasher.hexdigest()

    def train(self, json_path: Union[str, list]) -> Dict:
        r''' Train or continue training a model give a json_path containing a MedCATtrainer export. It will
        continue training if an existing model is loaded or start new training if the model is blank/new.

        Args:
            json_path (`str` or `list`):
                Path/Paths to a MedCATtrainer export containing the meta_annotations we want to train for.
            train_arguments(`str`, optional, defaults to `None`):
                HF TrainingArguments. If None the default will be used
        '''

        # Load the medcattrainer export
        if isinstance(json_path, str):
            json_path = [json_path]

        def merge_data_loaded(base, other):
            if not base:
                return other
            elif other is None:
                return base
            else:
                for p in other['projects']:
                    base['projects'].append(p)
            return base

        # Merge data from all different data paths
        data_loaded: Dict = {}
        for path in json_path:
            with open(path, 'r') as f:
                data_loaded = merge_data_loaded(data_loaded, json.load(f))

        # Here we have to save the data because of the data loader
        os.makedirs('results', exist_ok=True)
        json.dump(data_loaded, open("./results/data.json", 'w'))

        # Load dataset
        data_abs_path = os.path.join(os.getcwd(), 'results', 'data.json')
        dataset = datasets.load_dataset(os.path.abspath(transformers_ner.__file__),
                                        data_files={'train': data_abs_path},
                                        split=datasets.Split.TRAIN,
                                        cache_dir='/tmp/')

        # Update labelmap in case the current dataset has more labels than what we had before
        self.tokenizer.calculate_label_map(dataset)
        self.model.config.id2label = {v:k for k,v in self.tokenizer.label_map.items()}
        self.model.config.label2id = self.tokenizer.label_map

        if self.model.num_labels != len(self.tokenizer.label_map):
            self.log.warning("The dataset contains labels we've not seen before, model is being reinitialized")
            self.log.warning("Model: {} vs Dataset: {}".format(self.model.num_labels, len(self.tokenizer.label_map)))
            self.model = AutoModelForTokenClassification.from_pretrained(self.config.general['model_name'], num_labels=len(self.tokenizer.label_map))
            self.tokenizer.cui2name = {k:self.cdb.get_name(k) for k in self.tokenizer.label_map.keys()}

            self.model.config

        # Encode dataset
        encoded_dataset = dataset.map(
                lambda examples: self.tokenizer.encode(examples, ignore_subwords=False),
                batched=True,
                remove_columns=['ent_cuis', 'ent_ends', 'ent_starts', 'text'])
        # Split to train/test
        encoded_dataset = encoded_dataset.train_test_split(test_size=self.config.train.test_size)

        data_collator = CollateAndPadNER(self.tokenizer.hf_tokenizer.pad_token_id)
        trainer = Trainer(
                model = self.model,
                args=self.config.train,
                train_dataset=encoded_dataset['train'],
                eval_dataset=encoded_dataset['test'],
                compute_metrics=lambda p: metrics(p, tokenizer=self.tokenizer, dataset=encoded_dataset['test']),
                data_collator=data_collator,
                tokenizer=None)

        trainer.train()

        # Save the training time
        self.config.train.last_train_on = datetime.now().timestamp()

        # Save everything
        #self.save(save_dir_path=self.train.output_dir)

        # Run an eval step and return metrics
        p = trainer.predict(encoded_dataset['test'])
        df, examples = metrics(p, return_df=True, tokenizer=self.tokenizer, dataset=encoded_dataset['test'])

        # Create the pipeline for eval
        self.create_eval_pipeline()

        return df, examples

    def eval(self, json_path: str) -> Dict:
        raise Exception("Not Implemented!")

    def save(self, save_dir_path: str) -> None:
        r''' Save all components of this class to a file

        Args:
            save_dir_path(`str`):
                Path to the directory where everything will be saved.
        '''
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
    def load(cls, save_dir_path: str, config_dict: Optional[Dict] = None) -> "MetaCAT":
        r''' Load a meta_cat object.

        Args:
            save_dir_path (`str`):
                The directory where all was saved.
            config_dict (`dict`):
                This can be used to overwrite saved parameters for this meta_cat
                instance. Why? It is needed in certain cases where we autodeploy stuff.

        Returns:
            meta_cat (`medcat.MetaCAT`):
                You don't say
        '''

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
        r''' Process many documents at once.

        Args:
            stream (Iterable[spacy.tokens.Doc]):
                List of spacy documents.
        '''
        # Just in case
        if stream is None or not stream:
            return stream

        batch_size_chars = self.config.general['pipe_batch_size_in_chars']
        yield from self._process(stream, batch_size_chars)

    def _process(self,
                 stream: Iterable[Union[Doc, None]],
                 batch_size_chars: int) -> Iterator[Optional[Doc]]:
        for docs in self.batch_generator(stream, batch_size_chars):
            try:
                #all_text = [doc.text for doc in docs]
                #all_text_processed = self.tokenizer.encode_eval(all_text)
                # For now we will process the documents one by one, should be improved in the future to use batching
                for doc in docs:
                    res = self.ner_pipe(doc.text, aggregation_strategy=self.config.general['ner_aggregation_strategy'])
                    print(res)
                    for r in res:
                        inds = []
                        for ind, word in enumerate(doc):
                            end_char = word.idx + len(word.text)
                            if end_char <= r['end'] and end_char > r['start']:
                                inds.append(ind)
                            # To not loop through everything
                            if end_char > r['end']:
                                break

                        entity = Span(doc, min(inds), max(inds) + 1, label=r['entity_group'])
                        entity._.cui = r['entity_group']
                        entity._.context_similarity = r['score']
                        entity._.detected_name = r['word']
                        entity._.id = len(doc._.ents)
                        entity._.confidence = r['score']

                        doc._.ents.append(entity)

                yield from docs
            except Exception as e:
                # TODO: Make nice
                self.log.warning("Error in pipe: " + str(e))
                yield from [None] * len(docs)

    # Override
    def __call__(self, doc: Doc) -> Doc:
        ''' Process one document, used in the spacy pipeline for sequential
        document processing.

        Args:
            doc (spacy.tokens.Doc):
                A spacy document
        '''

        # Just call the pipe method
        doc = next(self.pipe(iter([doc])))

        return doc
