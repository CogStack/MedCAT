import os
import json
import pickle
import numpy as np
import torch
from scipy.special import softmax

from medcat.utils.ml_utils import train_network, eval_network
from medcat.utils.data_utils import prepare_from_json, encode_category_values, set_all_seeds
from medcat.preprocessing.tokenizers import TokenizerWrapperBPE
from medcat.preprocessing.tokenizers import TokenizerWrapperBERT
from medcat.config_meta_cat import ConfigMetaCAT


class MetaCAT(object):
    r''' TODO: Add documentation
    '''

    # Custom pipeline component name
    name = 'meta_cat'

    def __init__(self, tokenizer=None, embeddings=None, config=None):
        set_all_seeds(config.general['seed'])
        self.tokenizer = tokenizer
        if config is not None:
            self.config = config
        else:
            self.config = ConfigMetaCAT()
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32) if embeddings is not None else None

        self.model = self.get_model()

        if tokenizer is not None:
            # Set it in the config
            config.general['tokenizer_name'] = tokenizer.name

    def get_model(self):
        config = self.config
        model = None
        if config.model['model_name'] == 'lstm':
            from medcat.utils.models import LSTM
            model = LSTM(self.embeddings, config.model)

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
        data = json.load(open(json_path, 'r'))

        # Create directories if they don't exist
        if t_config['auto_save_model']:
            if save_dir_path is None:
                raise Exception("The `save_dir_path` argument is required if `aut_save_model` is `True` in the config")
            else:
                os.makedirs(save_dir_path, exist_ok=True)

        # Prepare the data
        data = prepare_from_json(data, g_config['cntx_left'], g_config['cntx_right'], self.tokenizer, cui_filter=t_config['cui_filter'],
                replace_center=g_config['replace_center'], cntx_in_chars=g_config['cntx_in_chars'], prerequisites=t_config['prerequisites'],
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

        (f1, p, r, cls_report) = train_network(self.model, data=data, config=self.config, save_dir_path=save_dir_path)

        # If autosave, then load the best model here
        if t_config['auto_save_model']:
            path = os.path.join(save_dir_path, 'model.dat')
            device = torch.device(g_config['device'])
            self.model.load_state_dict(torch.load(path, map_location=device))

        return {'f1':f1, 'p':p, 'r':r, 'cls_report': cls_report}


    def eval(self, json_path):
        g_config = self.config.general
        t_config = self.config.train

        data = json.load(open(json_path, 'r'))

        # Prepare the data
        data = prepare_from_json(data, g_config['cntx_left'], g_config['cntx_right'], self.tokenizer, cui_filter=t_config['cui_filter'],
                replace_center=g_config['replace_center'], cntx_in_chars=g_config['cntx_in_chars'], prerequisites=t_config['prerequisites'],
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
        result = eval_network(self.model, data, config=self.config)

        return result


    def predict_one(self, text, start, end):
        """ A test function, not useful in any other case
        """
        # Always cpu
        config = self.config
        device = torch.device('cpu')
        if config.general['lowercase']:
            text = text.lower()

        doc_text = self.tokenizer(text)
        ind = 0
        for ind, pair in enumerate(doc_text['offset_mapping']):
            if start >= pair[0] and start <= pair[1]:
                break
        cntx_left = config.general['cntx_left']
        cntx_right = config.general['cntx_right']
        _start = max(0, ind - cntx_left)
        _end = min(len(doc_text['tokens']), ind + 1 + cntx_right)
        tkns = doc_text['input_ids'][_start:_end]
        cpos = cntx_left + min(0, ind-cntx_left)

        x = torch.tensor([tkns], dtype=torch.long).to(device)
        cpos = torch.tensor([cpos], dtype=torch.long).to(device)

        self.model.eval()
        outputs_test = self.model(x, cpos)

        category_value2id = config.general['category_value2id']
        inv_map = {v: k for k, v in category_value2id.items()}
        return inv_map[int(np.argmax(outputs_test.detach().to('cpu').numpy()[0]))]


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

        # Save embeddings
        np.save(os.path.join(save_dir_path, 'embeddings.npy'), np.array(self.embeddings))

        # Save the model
        model_save_path = os.path.join(save_dir_path, 'model.dat')
        torch.save(self.model.state_dict(), model_save_path)

        # This is everything we need to save from the class, we do not
        #save the class itself.


    @classmethod
    def load(cls, save_dir_path):
        # Load config
        config = ConfigMetaCAT.load(os.path.join(save_dir_path, 'config.json'))

        # Load tokenizer
        if config.general['tokenizer_name'] == 'bbpe':
            from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBPE
            tokenizer = TokenizerWrapperBPE.load(save_dir_path)

        # Load embeddings
        embeddings = np.load(os.path.join(save_dir_path, 'embeddings.npy'))

        # Create meta_cat
        meta_cat = cls(tokenizer=tokenizer, embeddings=embeddings, config=config)

        # Load the model
        model_save_path = os.path.join(save_dir_path, 'model.dat')
        device = torch.device(config.general['device'])
        meta_cat.model.load_state_dict(torch.load(model_save_path, map_location=device))

        return meta_cat


    def __call__(self, doc):
        """ Spacy pipe method """
        config = self.config
        device = torch.device(config.general['device'])

        data = []
        id2row = {}
        text = doc.text
        if config.general['lowercase']:
            text = text.lower()
        doc_text = self.tokenizer(text)
        x = []
        cpos = []
        # Needed later

        id2category_value = {v: k for k, v in self.config.general['category_value2id'].items(l)}

        # Only loop through non overlapping entities
        for ent in doc.ents:
            start = ent.start_char
            end = ent.end_char
            ind = 0
            for ind, pair in enumerate(doc_text['offset_mapping']):
                if start >= pair[0] and start <= pair[1]:
                    break
            _start = max(0, ind - self.cntx_left)
            _end = min(len(doc_text['tokens']), ind + 1 + self.cntx_right)
            _ids = doc_text['input_ids'][_start:_end]
            _cpos = self.cntx_left + min(0, ind-self.cntx_left)

            id2row[ent._.id] = len(x)
            x.append(_ids)
            cpos.append(_cpos)

        max_seq_len = (self.cntx_left+self.cntx_right+1)
        x = np.array([(sample + [self.pad_id] * max(0, max_seq_len - len(sample)))[0:max_seq_len]
                      for sample in x])

        x = torch.tensor(x, dtype=torch.long).to(device)
        cpos = torch.tensor(cpos, dtype=torch.long).to(device)

        # Nearly impossible that we need batches, so I'll ignore it
        if len(x) >  0:
            self.model.eval()
            outputs = self.model(x, cpos).detach().to('cpu').numpy()
            confidences = softmax(outputs, axis=1)
            outputs = np.argmax(outputs, axis=1)

            for ent in doc.ents:
                val = id2category_value[outputs[id2row[ent._.id]]]
                confidence = confidences[id2row[ent._.id], outputs[id2row[ent._.id]]]
                if ent._.meta_anns is None:
                    ent._.meta_anns = {self.category_name: {'value': val,
                                                            'confidence': confidence,
                                                            'name': self.category_name}}
                else:
                    ent._.meta_anns[self.category_name] = {'value': val,
                                                           'confidence': confidence,
                                                           'name': self.category_name}

        return doc
