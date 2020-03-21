import json
from medcat.utils.data_utils import prepare_from_json, encode_category_values, tkns_to_ids
import torch
from medcat.utils.ml_utils import train_network
import numpy as np
import pickle

class MetaCAT(object):

    def __init__(self, tokenizer, embeddings, cntx_left=20, cntx_right=20,
                 save_dir='./meta_cat/', pad_id=30000):
        self.tokenizer = tokenizer
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.cntx_left = cntx_left
        self.cntx_right = cntx_right
        self.save_dir = save_dir
        self.pad_id = pad_id

        self.category_name = None
        self.category_values = {}
        self.i_category_values = {}

        self.model = None


    def train(self, json_path, category_name, model_name='lstm', lr=0.01, test_size=0.1,
              batch_size=100, nepochs=20, device='cpu', lowercase=True, class_weights=None, cv=0,
              ignore_cpos=False, model_config={}, tui_filter=None):
        data = json.load(open(json_path, 'r'))

        # Prepare the data
        data = prepare_from_json(data, self.cntx_left, self.cntx_right, self.tokenizer, lowercase=lowercase, tui_filter=tui_filter)

        # Check is the name there
        if category_name not in data:
            raise Exception("The category name does not exist in this json file")

        data = data[category_name]

        # Encode the category values
        self.category_name = category_name
        data, self.category_values = encode_category_values(data)
        self.i_category_values = {v: k for k, v in self.category_values.items()}

        # Convert data tkns to ids
        data = tkns_to_ids(data, self.tokenizer)

        if model_name == 'lstm':
            from medcat.utils.models import LSTM
            nclasses = len(self.category_values)
            bid = model_config.get("bid", True)
            print(bid)
            num_layers = model_config.get("num_layers", 2)
            input_size = model_config.get("input_size", 300)
            hidden_size = model_config.get("hidden_size", 300)
            dropout = model_config.get("dropout", 0.5)

            model = LSTM(self.embeddings, self.pad_id, nclasses=nclasses, bid=bid, num_layers=num_layers,
                         input_size=input_size, hidden_size=hidden_size, dropout=dropout)

        if cv == 0:
            (f1, p, r) = train_network(model, data, max_seq_len=(self.cntx_left+self.cntx_right+1), lr=lr, test_size=test_size,
                    pad_id=self.pad_id, batch_size=batch_size, nepochs=nepochs, device=device,
                    class_weights=class_weights, ignore_cpos=ignore_cpos)
        elif cv > 0:
            # Mainly for testing, not really used in a normal workflow
            f1s = []
            ps = []
            rs = []
            for i in range(cv):
                # Reset the model
                if model_name == 'lstm':
                    from medcat.utils.models import LSTM
                    nclasses = len(self.category_values)
                    model = LSTM(self.embeddings, self.pad_id, nclasses=nclasses)

                (_f1, _p, _r) = train_network(model, data, max_seq_len=(self.cntx_left+self.cntx_right+1), lr=lr, test_size=test_size,
                        pad_id=self.pad_id, batch_size=batch_size, nepochs=nepochs, device=device,
                        class_weights=class_weights, ignore_cpos=ignore_cpos)
                f1s.append(_f1)
                ps.append(_p)
                rs.append(_r)
            f1 = np.average(f1s)
            p = np.average(ps)
            r = np.average(rs)

        print("Best/Average scores: F1: {}, P: {}, R: {}".format(f1, p, r))

        self.model = model
        return {'f1':f1, 'p':p, 'r':r}


    def predicit_one(self, text, start, end):
        """ A test function, not useful in any other case
        """
        text = text.lower()

        doc_text = self.tokenizer.encode(text)
        ind = 0
        for ind, pair in enumerate(doc_text.offsets):
            if start >= pair[0] and start <= pair[1]:
                break
        _start = max(0, ind - self.cntx_left)
        _end = min(len(doc_text.tokens), ind + 1 + self.cntx_right)
        tkns = doc_text.ids[_start:_end]
        cpos = self.cntx_left + min(0, ind-self.cntx_left)

        device = torch.device("cpu")
        x = torch.tensor([tkns], dtype=torch.long).to(device)
        cpos = torch.tensor([cpos], dtype=torch.long).to(device)

        self.model.eval()
        outputs_test = self.model(x, cpos)

        inv_map = {v: k for k, v in self.category_values.items()}
        return inv_map[int(np.argmax(outputs_test.detach().numpy()[0]))]


    def save(self):
        # The model is saved during training, don't do it here
        #only save the config.
        self.save_config()


    def save_config(self):
        path = self.save_dir + "vars.dat"
        to_save = {'category_name': self.category_name,
                   'category_values': self.category_values,
                   'i_category_values': self.i_category_values,
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


    def load(self, model='lstm'):
        """ Loads model and config for this meta annotation
        """
        self.load_config()
        # Load MODEL
        if model == 'lstm':
            from medcat.utils.models import LSTM
            self.model = LSTM(self.embeddings, self.pad_id)
            path = self.save_dir + "lstm.dat"

        self.model.load_state_dict(torch.load(path))


    def __call__(self, doc, lowercase=True):
        """ Spacy pipe method """
        data = []
        id2row = {}
        text = doc.text
        if lowercase:
            text = text.lower()
        doc_text = self.tokenizer.encode(text)
        x = []
        cpos = []

        # Only loop through non overlapping entities
        for ent in doc.ents:
            start = ent.start_char
            end = ent.end_char
            ind = 0
            for ind, pair in enumerate(doc_text.offsets):
                if start >= pair[0] and start <= pair[1]:
                    break
            _start = max(0, ind - self.cntx_left)
            _end = min(len(doc_text.tokens), ind + 1 + self.cntx_left)
            _ids = doc_text.ids[_start:_end]
            _cpos = self.cntx_left + min(0, ind-self.cntx_left)

            id2row[ent._.id] = len(x)
            x.append(_ids)
            cpos.append(_cpos)

        max_seq_len = (self.cntx_left+self.cntx_right+1)
        x = np.array([(sample + [self.pad_id] * max(0, max_seq_len - len(sample)))[0:max_seq_len]
                      for sample in x])

        device = torch.device("cpu")
        x = torch.tensor(x, dtype=torch.long).to(device)
        cpos = torch.tensor(cpos, dtype=torch.long).to(device)

        # Nearly impossible that we need batches, so I'll ignore it
        if len(x) >  0:
            self.model.eval()
            outputs = self.model(x, cpos).detach().numpy()
            outputs = np.argmax(outputs, axis=1)

            for ent in doc.ents:
                val = self.i_category_values[outputs[id2row[ent._.id]]]
                if ent._.meta_anns is None:
                    ent._.meta_anns = {self.category_name: val}
                else:
                    ent._.meta_anns[self.category_name] = val

        return doc
