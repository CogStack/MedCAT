import json
from medcat.utils.data_utils import prepare_from_json, encode_category_values, tkns_to_ids
import torch
from medcat.utils.ml_utils import train_network

class MetaCAT(object):
    def __init__(self, cntx_size, tokenizer, vocab_size=30000, save_dir='./meta_cat/'):
        self.tokenizer = tokenizer
        self.cntx_size = cntx_size
        self.save_dir = save_dir

        self.category_name = None
        self.category_values = []

        self.vocab_size=vocab_size

        self.model = None
        self.embeddings = None

    def train(self, json_path, category_name, embeddings, model='lstm', pad_id=30000, max_seq_len=20):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.embeddings = embeddings
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
            model = LSTM(embeddings, pad_id)

        train_network(model, data, max_seq_len=max_seq_len)
        self.model = model

    def predict(self):
        pass


    def save(self):
        self.save_dict()


    def save_dict(self)
        path = self.save_dir + "vars.dat"
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)


    def _load_dict(self)
        """ Loads variables of this object
        """
        path = self.save_dir + "vars.dat"
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)


    def load(self):
        """ Loads model and variables
        """
        _load_dict()
        # Load MODEL
