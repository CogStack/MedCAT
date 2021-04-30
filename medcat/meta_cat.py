import os
import json
import pickle
import numpy as np
import torch
from scipy.special import softmax

from medcat.utils.ml_utils import train_network, eval_network
from medcat.utils.data_utils import prepare_from_json, encode_category_values, tkns_to_ids, set_all_seeds
from medcat.preprocessing.tokenizers import TokenizerWrapperBPE

class MetaCAT(object):
    r''' TODO: Add documentation
    '''
    def __init__(self, tokenizer=None, embeddings=None, cntx_left=20, cntx_right=20,
                 save_dir='./meta_cat/', pad_id=30000, device='cpu'):
        self.tokenizer = tokenizer
        if embeddings is not None:
            self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        else:
            self.embeddings = None
        self.cntx_left = cntx_left
        self.cntx_right = cntx_right
        self.save_dir = save_dir
        self.pad_id = pad_id
        self.device = torch.device(device)


        self.category_name = None
        self.category_values = {}
        self.i_category_values = {}

        self.model = None

        # TODO: A shitty solution, make right at some point
        if not self.save_dir.endswith("/"):
            self.save_dir = self.save_dir + "/"


    def train(self, json_path, category_name=None, model_name='lstm', lr=0.01, test_size=0.1,
              batch_size=100, nepochs=20, class_weights=None, cv=0,
              ignore_cpos=False, model_config={}, cui_filter=None, fine_tune=False,
              auto_save_model=True, score_average='weighted', replace_center=None, seed=11,
              prerequisite={}):
        r''' TODO: Docs
        '''
        set_all_seeds(seed)
        data = json.load(open(json_path, 'r'))

        # Create directories if they don't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Prepare the data
        data = prepare_from_json(data, self.cntx_left, self.cntx_right, self.tokenizer, cui_filter=cui_filter,
                replace_center=replace_center, cntx_in_chars=True, prerequisite=prerequisite)

        if category_name is not None:
            self.category_name = category_name

        # Check is the name there
        if self.category_name not in data:
            raise Exception("The category name does not exist in this json file. You've provided '{}', while the possible options are: {}".format(
                self.category_name, " | ".join(list(data.keys()))))

        data = data[self.category_name]

        if not fine_tune:
            # Encode the category values
            data, self.category_values = encode_category_values(data)
            self.i_category_values = {v: k for k, v in self.category_values.items()}
        else:
            # We already have everything, just get the data
            data, _ = encode_category_values(data, vals=self.category_values)

        # Convert data tkns to ids
        #data = tkns_to_ids(data, self.tokenizer)

        if not fine_tune:
            if model_name == 'lstm':
                from medcat.utils.models import LSTM
                nclasses = len(self.category_values)
                bid = model_config.get("bid", True)
                num_layers = model_config.get("num_layers", 2)
                input_size = model_config.get("input_size", 300)
                hidden_size = model_config.get("hidden_size", 300)
                dropout = model_config.get("dropout", 0.5)

                self.model = LSTM(self.embeddings, self.pad_id, nclasses=nclasses, bid=bid, num_layers=num_layers,
                             input_size=input_size, hidden_size=hidden_size, dropout=dropout)

        if cv == 0:
            (f1, p, r, cls_report) = train_network(self.model, data, max_seq_len=(self.cntx_left+self.cntx_right+1), lr=lr, test_size=test_size,
                    pad_id=self.pad_id, batch_size=batch_size, nepochs=nepochs, device=self.device,
                    class_weights=class_weights, ignore_cpos=ignore_cpos, save_dir=self.save_dir,
                    auto_save_model=auto_save_model, score_average=score_average)
        elif cv > 0:
            # Mainly for testing, not really used in a normal workflow
            f1s = []
            ps = []
            rs = []
            cls_reports = []
            for i in range(cv):
                # Reset the model
                if fine_tune:
                    self.load_model(model=model_name)
                else:
                    if model_name == 'lstm':
                        from medcat.utils.models import LSTM
                        nclasses = len(self.category_values)
                        self.model = LSTM(self.embeddings, self.pad_id, nclasses=nclasses)

                (_f1, _p, _r, _cls_report) = train_network(self.model, data, max_seq_len=(self.cntx_left+self.cntx_right+1), lr=lr, test_size=test_size,
                        pad_id=self.pad_id, batch_size=batch_size, nepochs=nepochs, device=self.device,
                        class_weights=class_weights, ignore_cpos=ignore_cpos, save_dir=self.save_dir, score_average=score_average)
                f1s.append(_f1)
                ps.append(_p)
                rs.append(_r)
                cls_reports.append(_cls_report)
            f1 = np.average(f1s)
            p = np.average(ps)
            r = np.average(rs)

            # Average cls reports
            cls_report = {}
            _cls_report = cls_reports[0]
            for label in _cls_report.keys():
                cls_report[label] = {}
                if type(_cls_report[label]) == dict:
                    for score in _cls_report[label].keys():
                        cls_report[label][score] = sum([r[label][score] for r in cls_reports]) / len(cls_reports)


        print("Best/Average scores: F1: {}, P: {}, R: {}".format(f1, p, r))

        return {'f1':f1, 'p':p, 'r':r, 'cls_report': cls_report}


    def eval(self, json_path, batch_size=100, lowercase=True, ignore_cpos=False, cui_filter=None, score_average='weighted',
            replace_center=None):
        data = json.load(open(json_path, 'r'))

        # Prepare the data
        data = prepare_from_json(data, self.cntx_left, self.cntx_right, self.tokenizer, lowercase=lowercase, cui_filter=cui_filter,
                replace_center=replace_center)

        # Check is the name there
        if self.category_name not in data:
            raise Exception("The category name does not exist in this json file.")

        data = data[self.category_name]

        # We already have everything, just get the data
        data, _ = encode_category_values(data, vals=self.category_values)

        # Convert data tkns to ids
        data = tkns_to_ids(data, self.tokenizer)

        # Run evaluation
        result = eval_network(self.model, data, max_seq_len=(self.cntx_left+self.cntx_right+1), pad_id=self.pad_id,
                batch_size=batch_size, device=self.device, ignore_cpos=ignore_cpos, score_average=score_average)

        return result


    def predicit_one(self, text, start, end):
        """ A test function, not useful in any other case
        """
        text = text.lower()

        doc_text = self.tokenizer(text)
        ind = 0
        for ind, pair in enumerate(doc_text['offset_mapping']):
            if start >= pair[0] and start <= pair[1]:
                break
        _start = max(0, ind - self.cntx_left)
        _end = min(len(doc_text['tokens']), ind + 1 + self.cntx_right)
        tkns = doc_text['input_ids'][_start:_end]
        cpos = self.cntx_left + min(0, ind-self.cntx_left)

        x = torch.tensor([tkns], dtype=torch.long).to(self.device)
        cpos = torch.tensor([cpos], dtype=torch.long).to(self.device)

        self.model.eval()
        outputs_test = self.model(x, cpos)

        inv_map = {v: k for k, v in self.category_values.items()}
        return inv_map[int(np.argmax(outputs_test.detach().to('cpu').numpy()[0]))]


    def save(self, full_save=False):
        if full_save:
            # Save tokenizer and embeddings, slightly redundant
            if hasattr(self.tokenizer, 'save_model'):
                # Support the new save in tokenizer 0.8.2+ from huggingface
                self.tokenizer.save_model(self.save_dir, name='bbpe')
            elif hasattr(self.tokenizer, 'save'):
                # The tokenizer wrapper saving  
                self.tokenizer.save(self.save_dir, name='bbpe')
            # Save embeddings
            np.save(open(self.save_dir + "embeddings.npy", 'wb'), np.array(self.embeddings))

        # The lstm model is saved during training, don't do it here
        #save the config.
        self.save_config()


    def save_config(self):
        # TODO: Add other parameters, e.g replace_center, ignore_cpos etc.
        path = self.save_dir + "vars.dat"
        to_save = {'category_name': self.category_name,
                   'category_values': self.category_values,
                   'i_category_values': self.i_category_values,
                   'pad_id': self.pad_id,
                   'cntx_left': self.cntx_left,
                   'cntx_right': self.cntx_right}
        with open(path, 'wb') as f:
            pickle.dump(to_save, f)


    def load_config(self):
        """ Loads variables of this object
        """
        path = self.save_dir + "vars.dat"
        with open(path, 'rb') as f:
            to_load = pickle.load(f)

        self.category_name = to_load['category_name']
        self.category_values = to_load['category_values']
        self.i_category_values = to_load['i_category_values']
        self.cntx_left = to_load['cntx_left']
        self.cntx_right = to_load['cntx_right']
        self.pad_id = to_load.get('pad_id', 0)


    def load_model(self, model='lstm'):
        # Load MODEL
        if model == 'lstm':
            from medcat.utils.models import LSTM
            nclasses = len(self.category_values)
            self.model = LSTM(self.embeddings, self.pad_id,
                              nclasses=nclasses)
            path = self.save_dir + "lstm.dat"

        self.model.load_state_dict(torch.load(path, map_location=self.device))


    def load(self, model='lstm', tokenizer_name='bbpe'):
        """ Loads model and config for this meta annotation
        """
        # Load tokenizer if it is None
        if self.tokenizer is None:
            self.tokenizer = TokenizerWrapperBPE.load(self.save_dir, name=tokenizer_name)
        # Load embeddings if None
        if self.embeddings is None:
            embeddings = np.load(open(self.save_dir  + "embeddings.npy", 'rb'), allow_pickle=False)
            self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

        # Load configuration
        self.load_config()

        # Load MODEL
        self.load_model(model=model)

    def __call__(self, doc, lowercase=True):
        """ Spacy pipe method """
        data = []
        id2row = {}
        text = doc.text
        if lowercase:
            text = text.lower()
        doc_text = self.tokenizer(text)
        x = []
        cpos = []

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

        x = torch.tensor(x, dtype=torch.long).to(self.device)
        cpos = torch.tensor(cpos, dtype=torch.long).to(self.device)

        # Nearly impossible that we need batches, so I'll ignore it
        if len(x) >  0:
            self.model.eval()
            outputs = self.model(x, cpos).detach().to('cpu').numpy()
            confidences = softmax(outputs, axis=1)
            outputs = np.argmax(outputs, axis=1)

            for ent in doc.ents:
                val = self.i_category_values[outputs[id2row[ent._.id]]]
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
