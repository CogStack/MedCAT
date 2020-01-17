import json
from medcat.utils.data_utils import prepare_from_json, encode_category_values, tkns_to_ids
import torch
from medcat.utils.ml_utils import train_network
import numpy as np
import pickle

class MetaCAT(object):
    def __init__(self, tokenizer, embeddings, cntx_size=20, vocab_size=30000,
                 save_dir='./meta_cat/'):
        self.tokenizer = tokenizer
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.cntx_size = cntx_size
        self.save_dir = save_dir

        self.category_name = None
        self.category_values = []

        self.vocab_size=vocab_size

        self.model = None

    def train(self, json_path, category_name, model='lstm',lr=0.01, test_size=0.1,
              pad_id=30000, batch_size=100, nepochs=20, device='cpu'):
        data = json.load(open(json_path, 'r'))

        # Prepare the data
        data = prepare_from_json(data, self.cntx_size, self.tokenizer)

        # Check is the name there
        if category_name not in data:
            raise Exception("The category name does not exist in this json file")

        data = data[category_name]

        # Encode the category values
        self.category_name = category_name
        data, self.category_values = encode_category_values(data)

        # Convert data tkns to ids
        data = tkns_to_ids(data, self.tokenizer)

        if model == 'lstm':
            from medcat.utils.models import LSTM
            model = LSTM(self.embeddings, pad_id)

        train_network(model, data, max_seq_len=(2*self.cntx_size+1), lr=lr, test_size=test_size,
                pad_id=pad_id, batch_size=batch_size, nepochs=nepochs, device=device)
        self.model = model

    def predict(self, text, annotations):
        """annotations: [(start, end), (start, end), ...]
        text: text of the document
        """
        pass


    def predicit_one(self, text, start, end):
        """ A test function, not useful in any other case
        """
        text = text.lower()

        doc_text = self.tokenizer.encode(text)
        ind = 0
        for ind, pair in enumerate(doc_text.offsets):
            if start >= pair[0] and start <= pair[1]:
                break
        _start = max(0, ind - self.cntx_size)
        _end = min(len(doc_text.tokens), ind + 1 + self.cntx_size)
        tkns = doc_text.ids[_start:_end]
        cpos = self.cntx_size + min(0, ind-self.cntx_size)

        device = torch.device("cpu")
        x = torch.tensor([tkns], dtype=torch.long).to(device)
        cpos = torch.tensor([cpos], dtype=torch.long).to(device)

        self.model.eval()
        outputs_test = self.model(x, cpos)

        inv_map = {v: k for k, v in self.category_values.items()}
        return inv_map[int(np.argmax(outputs_test.detach().numpy()[0]))]

    def save(self):
        self.save_config()


    def save_config(self):
        path = self.save_dir + "vars.dat"
        to_save = {'category_name': self.category_name,
                   'category_values': self.category_values,
                   'vocab_size': self.vocab_size,
                   'cntx_size': self.cntx_size}
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
        self.vocab_size = to_load['vocab_size']
        self.cntx_size = to_load['cntx_size']


    def load(self):
        """ Loads model and variables
        """
        load_config()
        # Load MODEL
