from typing import Dict, Any
from medcat.config import ConfigMixin


class ConfigRE(ConfigMixin):

    def __init__(self) -> None:

        self.general: Dict[str, Any] = {
                'device': 'cpu',
                'seed': 13,
                 # these are the pairs of relations that are to be predicted , form of : [("Disease", "Symptom"), ("entity1_type", "entity2_type") ...]
                'relation_type_filter_pairs': [],
                'category_value2id': {}, # Map from category values to ID, if empty it will be autocalculated during training
                'vocab_size': None, # Will be set automatically if the tokenizer is provided during meta_cat init
                'lowercase': True, # If true all input text will be lowercased
                'ent_context_left': 2, # Number of entities to take from the left of the concept
                'ent_context_right': 2, # Number of entities to take from the right of the concept
                'window_size' : 300, # max acceptable dinstance between entities (in characters)
                'batch_size_eval': 5000, # Number of annotations to be meta-annotated at once in eval
                'tokenizer_name': 'BERT', # Tokenizer name used with of MetaCAT
                'pipe_batch_size_in_chars': 20000000, # How many characters are piped at once into the meta_cat class
                }
        self.model: Dict[str, Any] = {
                'model_name': 'BERT',
                'input_size': 300,
                'hidden_size': 300,
                'dropout': 0.5,
                'nclasses': 2, # Number of classes that this model will output
                'padding_idx': -1,
                'emb_grad': True, # If True the embeddings will also be trained
                }

        self.train: Dict[str, Any] = {
                'batch_size': 100,
                'nepochs': 50,
                'lr': 0.001,
                'test_size': 0.1,
                'shuffle_data': True, # Used only during training, if set the dataset will be shuffled before train/test split
                'class_weights': None,
                'score_average': 'weighted', # What to use for averaging F1/P/R across labels
                'prerequisites': {},
                'auto_save_model': True, # Should do model be saved during training for best results
                }
