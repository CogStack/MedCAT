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
import datasets

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

        # Get the tokenizer if we do not have it 
        hf_tokenizer = AutoTokenizer.from_pretrained(self.config.general['model_name'])
        self.tokenizer = TransformersTokenizerNER(hf_tokenizer)

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

        if self.model.num_labels != len(self.tokenizer.label_map):
            self.log.warning("The dataset contains labels we've not seen before, model is being reinitialized")
            self.log.warning("Model: {} vs Dataset: {}".format(self.model.num_labels, len(self.tokenizer.label_map)))
            self.model = AutoModelForTokenClassification.from_pretrained(self.config.general['model_name'], num_labels=len(self.tokenizer.label_map))

        self.tokenizer.cui2name = {k:self.cdb.get_name(k) for k in self.tokenizer.label_map.keys()}

        # Encode dataset
        encoded_dataset = dataset.map(
                lambda examples: self.tokenizer.encode(examples, ignore_subwords=False),
                batched=True,
                remove_columns=['ent_cuis', 'ent_ends', 'ent_starts', 'text'])


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
        df, examples = metrics(p, return_df=True, tokenizer=tokenizer, dataset=encoded_dataset['test'])

        return df, examples

    def eval(self, json_path: str) -> Dict:
        """
        g_config = self.config.general
        t_config = self.config.train

        with open(json_path, 'r') as f:
            data_loaded: Dict = json.load(f)

        # Prepare the data
        assert self.tokenizer is not None
        data = prepare_from_json(data_loaded, g_config['cntx_left'], g_config['cntx_right'], self.tokenizer, cui_filter=t_config['cui_filter'],
                replace_center=g_config['replace_center'], prerequisites=t_config['prerequisites'],
                lowercase=g_config['lowercase'])

        # Check is the name there
        category_name = g_config['category_name']
        if category_name not in data:
            raise Exception("The category name does not exist in this json file.")

        data = data[category_name]

        # We already have everything, just get the data
        category_value2id = g_config['category_value2id']
        data, _ = encode_category_values(data, existing_category_value2id=category_value2id)

        # Run evaluation
        assert self.tokenizer is not None
        result = eval_model(self.model, data, config=self.config, tokenizer=self.tokenizer)

        return result
        """

    def save(self, save_dir_path: str) -> None:
        r''' Save all components of this class to a file

        Args:
            save_dir_path(`str`):
                Path to the directory where everything will be saved.
        '''
        # Create dirs if they do not exist
        os.makedirs(save_dir_path, exist_ok=True)

        # Save tokenizer
        assert self.tokenizer is not None
        self.tokenizer.save(save_dir_path)

        # Save config
        self.config.save(os.path.join(save_dir_path, 'config.json'))

        # Save the model
        model_save_path = os.path.join(save_dir_path, 'model.dat')
        torch.save(self.model.state_dict(), model_save_path)

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
        config = cast(ConfigTransformersNER, ConfigTransformersNER.load(os.path.join(save_dir_path, 'config.json')))

        # Overwrite loaded paramters with something new
        if config_dict is not None:
            config.merge_config(config_dict)

        tokenizer: Optional[TokenizerWrapperBase] = None
        # Load tokenizer (TODO: This should be converted into a factory or something)
        if config.general['tokenizer_name'] == 'bbpe':
            from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBPE
            tokenizer = TokenizerWrapperBPE.load(save_dir_path)
        elif config.general['tokenizer_name'] == 'bert-tokenizer':
            from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBERT
            tokenizer = TokenizerWrapperBERT.load(save_dir_path)

        # Create meta_cat
        meta_cat = cls(tokenizer=tokenizer, embeddings=None, config=config)

        # Load the model
        model_save_path = os.path.join(save_dir_path, 'model.dat')
        device = torch.device(config.general['device'])
        if not torch.cuda.is_available() and device.type == 'cuda':
            MetaCAT.log.warning('Loading a MetaCAT model without GPU availability, stored config used GPU')
            config.general['device'] = 'cpu'
            device = torch.device('cpu')
        meta_cat.model.load_state_dict(torch.load(model_save_path, map_location=device))

        return meta_cat

    def prepare_document(self, doc: Doc, input_ids: List, offset_mapping: List, lowercase: bool) -> Tuple:
        r'''

        Args:
            doc - spacy
            input_ids
            offset_mapping
        '''
        config = self.config
        cntx_left = config.general['cntx_left']
        cntx_right = config.general['cntx_right']
        replace_center = config.general['replace_center']

        # Should we annotate overlapping entities
        if config.general['annotate_overlapping']:
            ents = doc._.ents
        else:
            ents = doc.ents

        samples = []
        last_ind = 0
        ent_id2ind = {} # Map form entitiy ID to where is it in the samples array
        for ent in sorted(ents, key=lambda ent: ent.start_char):
            start = ent.start_char
            end = ent.end_char

            ind = 0
            # Start where the last ent was found, cannot be before it as we've sorted
            for ind, pair in enumerate(offset_mapping[last_ind:]):
                if start >= pair[0] and start < pair[1]:
                    break
            ind = last_ind + ind # If we did not start from 0 in the for loop
            last_ind = ind

            _start = max(0, ind - cntx_left)
            _end = min(len(input_ids), ind + 1 + cntx_right)
            tkns = input_ids[_start:_end]
            cpos = cntx_left + min(0, ind-cntx_left)

            if replace_center is not None:
                if lowercase:
                    replace_center = replace_center.lower()
                # We start from ind
                s_ind = ind
                e_ind = ind
                for _ind, pair in enumerate(offset_mapping[ind:]):
                    if end > pair[0] and end <= pair[1]:
                        e_ind = _ind + ind
                        break
                ln = e_ind - s_ind # Length of the concept in tokens
                assert self.tokenizer is not None
                tkns = tkns[:cpos] + self.tokenizer(replace_center)['input_ids'] + tkns[cpos+ln+1:]

            samples.append([tkns, cpos])
            ent_id2ind[ent._.id] = len(samples) - 1

        return ent_id2ind, samples

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

    # Override
    def pipe(self, stream: Iterable[Union[Doc, None]], *args, **kwargs) -> Iterator[Doc]:
        r''' Process many documents at once.

        Args:
            stream (Iterable[spacy.tokens.Doc]):
                List of spacy documents.
        '''
        # Just in case
        if stream is None or not stream:
            return stream

        config = self.config
        id2category_value = {v: k for k, v in config.general['category_value2id'].items()}
        batch_size_chars = config.general['pipe_batch_size_in_chars']

        if config.general['device'] == 'cpu' or config.general['disable_component_lock']:
            yield from self._set_meta_anns(stream, batch_size_chars, config, id2category_value)
        else:
            with MetaCAT._component_lock:
                yield from self._set_meta_anns(stream, batch_size_chars, config, id2category_value)

    def _set_meta_anns(self,
                       stream: Iterable[Union[Doc, None]],
                       batch_size_chars: int,
                       config: ConfigTransformersNER,
                       id2category_value: Dict) -> Iterator[Optional[Doc]]:
        for docs in self.batch_generator(stream, batch_size_chars):
            try:
                if not config.general['save_and_reuse_tokens'] or docs[0]._.share_tokens is None:
                    if config.general['lowercase']:
                        all_text = [doc.text.lower() for doc in docs]
                    else:
                        all_text = [doc.text for doc in docs]
                    assert self.tokenizer is not None
                    all_text_processed = self.tokenizer(all_text)
                    doc_ind2positions = {}
                    data: List = [] # The thing that goes into the model
                    for i, doc in enumerate(docs):
                        ent_id2ind, samples = self.prepare_document(doc, input_ids=all_text_processed[i]['input_ids'],
                                                                    offset_mapping=all_text_processed[i]['offset_mapping'],
                                                                    lowercase=config.general['lowercase'])
                        doc_ind2positions[i] = (len(data), len(data) + len(samples), ent_id2ind) # Needed so we know where is what in the big data array
                        data.extend(samples)
                        if config.general['save_and_reuse_tokens']:
                            doc._.share_tokens = (samples, doc_ind2positions[i])
                else:
                    # This means another model has already processed the data and we can just use it. This is a
                    #dangerous option - as it assumes the other model has the same tokenizer and context size.
                    data = []
                    doc_ind2positions = {}
                    for i, doc in enumerate(docs):
                        data.extend(doc._.share_tokens[0])
                        doc_ind2positions[i] = doc._.share_tokens[1]

                all_predictions, all_confidences = predict(self.model, data, config)
                for i, doc in enumerate(docs):
                    start_ind, end_ind, ent_id2ind = doc_ind2positions[i]

                    predictions = all_predictions[start_ind:end_ind]
                    confidences = all_confidences[start_ind:end_ind]
                    if config.general['annotate_overlapping']:
                        ents = doc._.ents
                    else:
                        ents = doc.ents

                    for ent in ents:
                        ent_ind = ent_id2ind[ent._.id]
                        value = id2category_value[predictions[ent_ind]]
                        confidence = confidences[ent_ind]
                        if ent._.meta_anns is None:
                            ent._.meta_anns = {config.general['category_name']: {'value': value,
                                                                                 'confidence': float(confidence),
                                                                                 'name': config.general['category_name']}}
                        else:
                            ent._.meta_anns[config.general['category_name']] = {'value': value,
                                                                                'confidence': float(confidence),
                                                                                'name': config.general['category_name']}
                yield from docs
            except Exception as e:
                self.get_error_handler()(self.name, self, docs, e)
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
