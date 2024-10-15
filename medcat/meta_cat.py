import os
import json
import logging
import torch
import numpy
from multiprocessing import Lock
from torch import nn, Tensor
from spacy.tokens import Doc, Span
from datetime import datetime
from typing import Iterable, Iterator, Optional, Dict, List, Tuple, cast, Union
from medcat.utils.hasher import Hasher
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.utils.meta_cat.ml_utils import predict, train_model, set_all_seeds, eval_model
from medcat.utils.meta_cat.data_utils import prepare_from_json, encode_category_values, prepare_for_oversampled_data
from medcat.pipeline.pipe_runner import PipeRunner
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBase
from medcat.utils.meta_cat.data_utils import Doc as FakeDoc
from peft import get_peft_model, LoraConfig, TaskType

# It should be safe to do this always, as all other multiprocessing
# will be finished before data comes to meta_cat
os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)  # separate logger from the package-level one


class MetaCAT(PipeRunner):
    """The MetaCAT class used for training 'Meta-Annotation' models, i.e. annotations of clinical
    concept annotations. These are also known as properties or attributes of recognise entities
    in similar tools such as MetaMap and cTakes.

    This is a flexible model agnostic class that can learns any meta-annotation task, i.e. any
    multi-class classification task for recognised terms.

    Args:
        tokenizer (TokenizerWrapperBase):
            The Huggingface tokenizer instance. This can be a pre-trained tokenzier instance from
            a BERT-style model, or trained from scratch for the Bi-LSTM (w. attention) model that
            is currently used in most deployments.
        embeddings (Tensor, numpy.ndarray):
            embedding mapping (sub)word input id n-dim (sub)word embedding.
        config (ConfigMetaCAT):
            the configuration for MetaCAT. Param descriptions available in ConfigMetaCAT docs.
    """

    # Custom pipeline component name
    name = 'meta_cat'
    _component_lock = Lock()

    # Override
    def __init__(self,
                 tokenizer: Optional[TokenizerWrapperBase] = None,
                 embeddings: Optional[Union[Tensor, numpy.ndarray]] = None,
                 config: Optional[ConfigMetaCAT] = None) -> None:
        if config is None:
            config = ConfigMetaCAT()
        self.config = config
        set_all_seeds(config.general['seed'])

        if tokenizer is not None:
            # Set it in the config
            config.general['tokenizer_name'] = tokenizer.name
            config.general['vocab_size'] = tokenizer.get_size()
            # We will also set the padding
            config.model['padding_idx'] = tokenizer.get_pad_id()
        self.tokenizer = tokenizer

        self.embeddings = torch.tensor(embeddings, dtype=torch.float32) if embeddings is not None else None
        self.model = self.get_model(embeddings=self.embeddings)

    def get_model(self, embeddings: Optional[Tensor]) -> nn.Module:
        """Get the model

        Args:
            embeddings (Optional[Tensor]):
                The embedding densor

        Raises:
            ValueError: If the meta model is not LSTM or BERT

        Returns:
            nn.Module:
                The module
        """
        config = self.config
        if config.model['model_name'] == 'lstm':
            from medcat.utils.meta_cat.models import LSTM
            model: nn.Module = LSTM(embeddings, config)
            logger.info("LSTM model used for classification")

        elif config.model['model_name'] == 'bert':
            from medcat.utils.meta_cat.models import BertForMetaAnnotation
            model = BertForMetaAnnotation(config)

            if not config.model.model_freeze_layers:
                peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=16,
                                         target_modules=["query", "value"], lora_dropout=0.2)

                model = get_peft_model(model, peft_config)
                # model.print_trainable_parameters()

            logger.info("BERT model used for classification")

        else:
            raise ValueError("Unknown model name %s" % config.model['model_name'])

        return model

    def get_hash(self) -> str:
        """A partial hash trying to catch differences between models.

        Returns:
            str: The hex hash.
        """
        hasher = Hasher()
        # Set last_train_on if None
        if self.config.train.last_train_on is None:
            self.config.train.last_train_on = datetime.now().timestamp()

        hasher.update(self.config.get_hash())
        return hasher.hexdigest()

    def train_from_json(self, json_path: Union[str, list], save_dir_path: Optional[str] = None,
                        data_oversampled: Optional[list] = None) -> Dict:
        """Train or continue training a model give a json_path containing a MedCATtrainer export. It will
        continue training if an existing model is loaded or start new training if the model is blank/new.

        Args:
            json_path (Union[str, list]):
                Path/Paths to a MedCATtrainer export containing the meta_annotations we want to train for.
            save_dir_path (Optional[str]):
                In case we have aut_save_model (meaning during the training the best model will be saved)
                we need to set a save path. Defaults to `None`.
            data_oversampled (Optional[list]):
                In case of oversampling being performed, the data will be passed in the parameter allowing the
                model to be trained on original + synthetic data.

        Returns:
            Dict: The resulting report.
        """

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
        return self.train_raw(data_loaded, save_dir_path, data_oversampled=data_oversampled)

    def train_raw(self, data_loaded: Dict, save_dir_path: Optional[str] = None, data_oversampled: Optional[list] = None) -> Dict:
        """
        Train or continue training a model given raw data. It will continue training if an existing model is loaded or start new training if the model is blank/new.

        The raw data is expected in the following format:

        {

            'projects': [  # list of projects

                {

                    'name': '<project_name>',

                    'documents': [  # list of documents

                        {

                            'name': '<document_name>',

                            'text': '<text_of_document>',

                            'annotations': [  # list of annotations

                                {

                                    'start': -1,  # start index of the annotation

                                    'end': 1,    # end index of the annotation

                                    'cui': 'cui',

                                    'value': '<annotation_value>'
                                },
                                ...
                            ],
                        },
                        ...
                    ]
                },
                ...
            ]
        }

        Args:
            data_loaded (Dict):
                The raw data we want to train for.
            save_dir_path (Optional[str]):
                In case we have aut_save_model (meaning during the training the best model will be saved)
                we need to set a save path. Defaults to `None`.
            data_oversampled (Optional[list]):
                In case of oversampling being performed, the data will be passed in the parameter allowing the
                model to be trained on original + synthetic data.
                The format of which is expected: [[['text','of','the','document'], [index of medical entity], "label" ],
                ['text','of','the','document'], [index of medical entity], "label" ]]

        Returns:
            Dict: The resulting report.

        Raises:
            Exception: If no save path is specified, or category name not in data.
            AssertionError: If no tokeniser is set
            FileNotFoundError: If phase_number is set to 2 and model.dat file is not found
            KeyError: If phase_number is set to 2 and model.dat file contains mismatched architecture
        """
        g_config = self.config.general
        t_config = self.config.train

        # Create directories if they don't exist
        if t_config['auto_save_model']:
            if save_dir_path is None:
                raise Exception("The `save_dir_path` argument is required if `aut_save_model` is `True` in the config")
            else:
                os.makedirs(save_dir_path, exist_ok=True)

        # Prepare the data
        assert self.tokenizer is not None
        data = prepare_from_json(data_loaded, g_config['cntx_left'], g_config['cntx_right'], self.tokenizer,
                                 cui_filter=t_config['cui_filter'],
                                 replace_center=g_config['replace_center'], prerequisites=t_config['prerequisites'],
                                 lowercase=g_config['lowercase'])

        # Check is the name present
        category_name = g_config['category_name']
        if category_name not in data:
            raise Exception(
                "The category name does not exist in this json file. You've provided '{}', while the possible options are: {}".format(
                    category_name, " | ".join(list(data.keys()))))

        data = data[category_name]
        if data_oversampled:
            data_sampled = prepare_for_oversampled_data(data_oversampled, self.tokenizer)
            data = data + data_sampled

        category_value2id = g_config['category_value2id']
        if not category_value2id:
            # Encode the category values
            full_data, data_undersampled, category_value2id = encode_category_values(data,
                                                                                     category_undersample=self.config.model.category_undersample)
            g_config['category_value2id'] = category_value2id
        else:
            # We already have everything, just get the data
            full_data, data_undersampled, category_value2id = encode_category_values(data,
                                                                                     existing_category_value2id=category_value2id,
                                                                                     category_undersample=self.config.model.category_undersample)
            g_config['category_value2id'] = category_value2id
        # Make sure the config number of classes is the same as the one found in the data
        if len(category_value2id) != self.config.model['nclasses']:
            logger.warning(
                "The number of classes set in the config is not the same as the one found in the data: %d vs %d",self.config.model['nclasses'], len(category_value2id))
            logger.warning("Auto-setting the nclasses value in config and rebuilding the model.")
            self.config.model['nclasses'] = len(category_value2id)

        if self.config.model.phase_number == 2 and save_dir_path is not None:
            model_save_path = os.path.join(save_dir_path, 'model.dat')
            device = torch.device(g_config['device'])
            try:
                self.model.load_state_dict(torch.load(model_save_path, map_location=device))
                logger.info("Model state loaded from dict for 2 phase learning")

            except FileNotFoundError:
                raise FileNotFoundError(f"\nError: Model file not found at path: {model_save_path}\nPlease run phase 1 training and then run phase 2.")

            except KeyError:
                raise KeyError("\nError: Missing key in loaded state dictionary. \nThis might be due to a mismatch between the model architecture and the saved state.")

            except Exception as e:
                raise Exception(f"\nError: Model state cannot be loaded from dict. {e}")

        data = full_data
        if self.config.model.phase_number == 1:
            data = data_undersampled
            if not t_config['auto_save_model']:
                logger.info("For phase 1, model state has to be saved. Saving model...")
                t_config['auto_save_model'] = True

        report = train_model(self.model, data=data, config=self.config, save_dir_path=save_dir_path)

        # If autosave, then load the best model here
        if t_config['auto_save_model']:
            if save_dir_path is None:
                raise Exception("The `save_dir_path` argument is required if `aut_save_model` is `True` in the config")
            else:
                path = os.path.join(save_dir_path, 'model.dat')
                device = torch.device(g_config['device'])
                self.model.load_state_dict(torch.load(path, map_location=device))

                # Save everything now
                self.save(save_dir_path=save_dir_path)

        self.config.train.last_train_on = datetime.now().timestamp()
        return report

    def eval(self, json_path: str) -> Dict:
        """Evaluate from json.

        Args:
            json_path (str):
                The json file ath

        Returns:
            Dict:
                The resulting model dict

        Raises:
            AssertionError: If self.tokenizer
            Exception: If the category name does not exist
        """
        g_config = self.config.general
        t_config = self.config.train

        with open(json_path, 'r') as f:
            data_loaded: Dict = json.load(f)

        # Prepare the data
        assert self.tokenizer is not None
        data = prepare_from_json(data_loaded, g_config['cntx_left'], g_config['cntx_right'], self.tokenizer,
                                 cui_filter=t_config['cui_filter'],
                                 replace_center=g_config['replace_center'], prerequisites=t_config['prerequisites'],
                                 lowercase=g_config['lowercase'])

        # Check is the name there
        category_name = g_config['category_name']
        if category_name not in data:
            raise Exception("The category name does not exist in this json file.")

        data = data[category_name]

        # We already have everything, just get the data
        category_value2id = g_config['category_value2id']
        data, _, _ = encode_category_values(data, existing_category_value2id=category_value2id)

        # Run evaluation
        assert self.tokenizer is not None
        result = eval_model(self.model, data, config=self.config, tokenizer=self.tokenizer)

        return result

    def save(self, save_dir_path: str) -> None:
        """Save all components of this class to a file

        Args:
            save_dir_path (str):
                Path to the directory where everything will be saved.

        Raises:
            AssertionError: If self.tokenizer is None
        """
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
        # save the class itself.

    @classmethod
    def load(cls, save_dir_path: str, config_dict: Optional[Dict] = None) -> "MetaCAT":
        """Load a meta_cat object.

        Args:
            save_dir_path (str):
                The directory where all was saved.
            config_dict (Optional[Dict], optional):
                This can be used to overwrite saved parameters for this meta_cat
                instance. Why? It is needed in certain cases where we autodeploy stuff. (Default value = None)

        Returns:
            MetaCAT:
                The MetaCAT instance
        """

        # Load config
        config = cast(ConfigMetaCAT, ConfigMetaCAT.load(os.path.join(save_dir_path, 'config.json')))

        # Overwrite loaded parameters with something new
        if config_dict is not None:
            config.merge_config(config_dict)

        tokenizer: Optional[TokenizerWrapperBase] = None
        # Load tokenizer (TODO: This should be converted into a factory or something)
        if config.general['tokenizer_name'] == 'bbpe':
            from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBPE
            tokenizer = TokenizerWrapperBPE.load(save_dir_path)
        elif config.general['tokenizer_name'] == 'bert-tokenizer':
            from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBERT
            tokenizer = TokenizerWrapperBERT.load(save_dir_path, config.model['model_variant'])

        # Create meta_cat
        meta_cat = cls(tokenizer=tokenizer, embeddings=None, config=config)

        # Load the model
        model_save_path = os.path.join(save_dir_path, 'model.dat')
        device = torch.device(config.general['device'])
        if not torch.cuda.is_available() and device.type == 'cuda':
            logger.warning('Loading a MetaCAT model without GPU availability, stored config used GPU')
            config.general['device'] = 'cpu'
            device = torch.device('cpu')
        meta_cat.model.load_state_dict(torch.load(model_save_path, map_location=device))

        return meta_cat

    def get_ents(self, doc: Doc) -> Iterable[Span]:
        spangroup_name = self.config.general.span_group
        if spangroup_name:
            try:
                return doc.spans[spangroup_name]
            except KeyError:
                raise Exception(
                    f"Configuration error MetaCAT was configured to set meta_anns on {spangroup_name} but this spangroup was not set on the doc.")

        # Should we annotate overlapping entities
        if self.config.general['annotate_overlapping']:
            return doc._.ents

        return doc.ents

    def prepare_document(self, doc: Doc, input_ids: List, offset_mapping: List, lowercase: bool) -> Tuple:
        """Prepares document.

        Args:
            doc (Doc):
                The document
            input_ids (List):
                Input ids
            offset_mapping (List):
                Offset mappings
            lowercase (bool):
                Whether to use lower case replace center

        Returns:
            Dict:
                Entity id to index mapping
            List:
                Samples
        """
        config = self.config
        cntx_left = config.general['cntx_left']
        cntx_right = config.general['cntx_right']
        replace_center = config.general['replace_center']

        ents = self.get_ents(doc)

        samples = []
        last_ind = 0
        ent_id2ind = {}  # Map form entity ID to where is it in the samples array
        for ent in sorted(ents, key=lambda ent: ent.start_char):
            start = ent.start_char
            end = ent.end_char

            # Updated implementation to extract all the tokens for the medical entity (rather than the one)
            ctoken_idx = []
            for ind, pair in enumerate(offset_mapping[last_ind:]):
                # Checking if we've reached at the start of the entity
                if start <= pair[0] or start <= pair[1]:
                    if end <= pair[1]:
                        ctoken_idx.append(ind) # End reached
                        break
                    else:
                        ctoken_idx.append(ind) # Keep going

            # Start where the last ent was found, cannot be before it as we've sorted
            last_ind += ind  # If we did not start from 0 in the for loop

            _start = max(0, ctoken_idx[0] - cntx_left)
            _end = min(len(input_ids), ctoken_idx[-1] + 1 + cntx_right)

            tkns = input_ids[_start:_end]
            cpos = cntx_left + min(0, ind - cntx_left)
            cpos_new = [x - _start for x in ctoken_idx]

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
                ln = e_ind - s_ind  # Length of the concept in tokens
                assert self.tokenizer is not None
                tkns = tkns[:cpos] + self.tokenizer(replace_center)['input_ids'] + tkns[cpos + ln + 1:]
            samples.append([tkns, cpos_new])
            ent_id2ind[ent._.id] = len(samples) - 1

        return ent_id2ind, samples

    @staticmethod
    def batch_generator(stream: Iterable[Doc], batch_size_chars: int) -> Iterable[List[Doc]]:
        """Generator for batch of documents.

        Args:
            stream (Iterable[Doc]):
                The document stream
            batch_size_chars (int):
                Number of characters per batch

        Yields:
            List[Doc]: The batch of documents.
        """
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
    def pipe(self, stream: Iterable[Union[Doc, FakeDoc]], *args, **kwargs) -> Iterator[Doc]:
        """Process many documents at once.

        Args:
            stream (Iterable[Union[Doc, FakeDoc]]):
                List of spacy documents.
            *args: Unused arguments (due to override)
            **kwargs: Unused keyword arguments (due to override)

        Yields:
            Doc: The document.

        Returns:
            Iterator[Doc]: stream is None or empty.
        """
        # Just in case
        if stream is None or not stream:
            # return an empty generator
            return

        config = self.config
        id2category_value = {v: k for k, v in config.general['category_value2id'].items()}
        batch_size_chars = config.general['pipe_batch_size_in_chars']

        if config.general['device'] == 'cpu' or config.general['disable_component_lock']:
            yield from self._set_meta_anns(stream, batch_size_chars, config, id2category_value)  # type: ignore
        else:
            with MetaCAT._component_lock:
                yield from self._set_meta_anns(stream, batch_size_chars, config, id2category_value)  # type: ignore

    def _set_meta_anns(self,
                       stream: Iterable[Union[Doc, FakeDoc]],
                       batch_size_chars: int,
                       config: ConfigMetaCAT,
                       id2category_value: Dict) -> Iterator[Optional[Doc]]:
        for docs in self.batch_generator(stream, batch_size_chars):  # type: ignore
            try:
                if not config.general['save_and_reuse_tokens'] or docs[0]._.share_tokens is None:
                    if config.general['lowercase']:
                        all_text = [doc.text.lower() for doc in docs]
                    else:
                        all_text = [doc.text for doc in docs]
                    assert self.tokenizer is not None
                    all_text_processed = self.tokenizer(all_text)
                    doc_ind2positions = {}
                    data: List = []  # The thing that goes into the model
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
                    # dangerous option - as it assumes the other model has the same tokenizer and context size.
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
                    ents = self.get_ents(doc)

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
        """Process one document, used in the spacy pipeline for sequential
        document processing.

        Args:
            doc (Doc):
                A spacy document

        Returns:
            Doc: The same spacy document.
        """

        # Just call the pipe method
        doc = next(self.pipe(iter([doc])))

        return doc

    def get_model_card(self, as_dict: bool = False) -> Union[str, dict]:
        """A minimal model card.

        Args:
            as_dict (bool):
                Return the model card as a dictionary instead of a str. (Default value = False)

        Returns:
            Union[str, dict]:
                An indented JSON object.
                OR A JSON object in dict form.
        """
        card = {
            'Category Name': self.config.general['category_name'],
            'Description': self.config.general['description'],
            'Classes': self.config.general['category_value2id'],
            'Model': self.config.model['model_name']
        }
        if as_dict:
            return card
        else:
            return json.dumps(card, indent=2, sort_keys=False)

    def __repr__(self):
        """Prints the model_card for this MetaCAT instance.

        Returns:
            the 'Model Card' for this MetaCAT instance. This includes NER+L config and any MetaCATs
        """
        return self.get_model_card(as_dict=False)
