import os
import json
import logging
import torch
from multiprocessing import Lock
from spacy.tokens import Doc
from typing import Iterable, Generator
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.utils.meta_cat.ml_utils import predict, train_model, set_all_seeds, eval_model
from medcat.utils.meta_cat.data_utils import prepare_from_json, encode_category_values
from medcat.utils.loggers import add_handlers
from medcat.pipeline.pipe_runner import PipeRunner

# It should be safe to do this always, as all other multiprocessing
#will be finished before data comes to meta_cat
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class MetaCAT(PipeRunner):
    r''' TODO: Add documentation
    '''

    # Custom pipeline component name
    name = 'meta_cat'
    _component_lock = Lock()

    log = logging.getLogger(__package__)
    # Add file and console handlers
    log = add_handlers(log)

    def __init__(self, tokenizer=None, embeddings=None, config=None):
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

    def get_model(self, embeddings):
        config = self.config
        model = None
        if config.model['model_name'] == 'lstm':
            from medcat.utils.meta_cat.models import LSTM
            model = LSTM(embeddings, config)

        return model

    def train(self, json_path, save_dir_path=None):
        r''' Train or continue training a model give a json_path containing a MedCATtrainer export. It will
        continue training if an existing model is loaded or start new training if the model is blank/new.

        Args:
            json_path (`str`):
                Path to a MedCATtrainer export containing the meta_annotations we want to train for.
            save_dir_path (`str`, optional, defaults to `None`):
                In case we have aut_save_model (meaning during the training the best model will be saved)
                we need to set a save path.
        '''
        g_config = self.config.general
        t_config = self.config.train

        # Load the medcattrainer export
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Create directories if they don't exist
        if t_config['auto_save_model']:
            if save_dir_path is None:
                raise Exception("The `save_dir_path` argument is required if `aut_save_model` is `True` in the config")
            else:
                os.makedirs(save_dir_path, exist_ok=True)

        # Prepare the data
        data = prepare_from_json(data, g_config['cntx_left'], g_config['cntx_right'], self.tokenizer, cui_filter=t_config['cui_filter'],
                replace_center=g_config['replace_center'], prerequisites=t_config['prerequisites'],
                lowercase=g_config['lowercase'])

        # Check is the name there
        category_name = g_config['category_name']
        if category_name not in data:
            raise Exception("The category name does not exist in this json file. You've provided '{}', while the possible options are: {}".format(
                category_name, " | ".join(list(data.keys()))))

        data = data[category_name]

        category_value2id = g_config['category_value2id']
        if not category_value2id:
            # Encode the category values
            data, category_value2id = encode_category_values(data)
            g_config['category_value2id'] = category_value2id
        else:
            # We already have everything, just get the data
            data, _ = encode_category_values(data, existing_category_value2id=category_value2id)

        # Make sure the config number of classes is the same as the one found in the data
        if len(category_value2id) != self.config.model['nclasses']:
            self.log.warning("The number of classes set in the config is not the same as the one found in the data: {} vs {}".format(
                             self.config.model['nclasses'], len(category_value2id)))
            self.log.warning("Auto-setting the nclasses value in config and rebuilding the model.")
            self.config.model['nclasses'] = len(category_value2id)
            self.model = self.get_model(embeddings=self.embeddings)

        report = train_model(self.model, data=data, config=self.config, save_dir_path=save_dir_path)

        # If autosave, then load the best model here
        if t_config['auto_save_model']:
            path = os.path.join(save_dir_path, 'model.dat')
            device = torch.device(g_config['device'])
            self.model.load_state_dict(torch.load(path, map_location=device))

            # Save everything now
            self.save(save_dir_path=save_dir_path)

        return report

    def eval(self, json_path):
        g_config = self.config.general
        t_config = self.config.train

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Prepare the data
        data = prepare_from_json(data, g_config['cntx_left'], g_config['cntx_right'], self.tokenizer, cui_filter=t_config['cui_filter'],
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
        result = eval_model(self.model, data, config=self.config, tokenizer=self.tokenizer)

        return result

    def save(self, save_dir_path):
        r''' Save all components of this class to a file

        Args:
            save_dir_path(`str`):
                Path to the directory where everything will be saved.
        '''
        # Create dirs if they do not exist
        os.makedirs(save_dir_path, exist_ok=True)

        # Save tokenizer
        self.tokenizer.save(save_dir_path)

        # Save config
        self.config.save(os.path.join(save_dir_path, 'config.json'))

        # Save the model
        model_save_path = os.path.join(save_dir_path, 'model.dat')
        torch.save(self.model.state_dict(), model_save_path)

        # This is everything we need to save from the class, we do not
        #save the class itself.

    @classmethod
    def load(cls, save_dir_path, config_dict=None):
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
        config = ConfigMetaCAT.load(os.path.join(save_dir_path, 'config.json'))

        # Overwrite loaded paramters with something new
        if config_dict is not None:
            config.merge_config(config_dict)

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
        meta_cat.model.load_state_dict(torch.load(model_save_path, map_location=device))

        return meta_cat

    def prepare_document(self, doc, input_ids, offset_mapping, lowercase):
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
                tkns = tkns[:cpos] + self.tokenizer(replace_center)['input_ids'] + tkns[cpos+ln+1:]

            samples.append([tkns, cpos])
            ent_id2ind[ent._.id] = len(samples) - 1

        return ent_id2ind, samples

    @staticmethod
    def batch_generator(stream, batch_size_chars):
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
    def pipe(self, stream: Iterable[Doc], *args, **kwargs) -> Generator[Doc, None, None]:
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

    def _set_meta_anns(self, stream, batch_size_chars, config, id2category_value):
        for docs in self.batch_generator(stream, batch_size_chars):
            try:
                if not config.general['save_and_reuse_tokens'] or docs[0]._.share_tokens is None:
                    if config.general['lowercase']:
                        all_text = [doc.text.lower() for doc in docs]
                    else:
                        all_text = [doc.text for doc in docs]

                    all_text_processed = self.tokenizer(all_text)
                    doc_ind2positions = {}
                    data = [] # The thing that goes into the model
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

    def __call__(self, doc):
        ''' Process one document, used in the spacy pipeline for sequential
        document processing.

        Args:
            doc (spacy.tokens.Doc):
                A spacy document
        '''

        # Just call the pipe method
        doc = next(self.pipe(iter([doc])))

        return doc
