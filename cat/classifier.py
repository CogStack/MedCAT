import torch
from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam, BertModel, BertTokenizer, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn
from torch.optim import Adam
import numpy as np
from sklearn.metrics import f1_score

class Classifier(object):
    def __init__(self, sequence_length, model='bert-base-uncased', LOWER_CASE=True, num_labels=2):
        self.config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                            num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
        self.sequence_length = sequence_length
        self.num_labels = num_labels
        self.model = BertModel(config)
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=LOWER_CASE)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def train(self, json_data):
        pass

    def predict(self, text):
        pass

    def predict_json(self, json_data):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
