from typing import Dict, Any
from medcat.config import ConfigMixin


class ConfigMetaCAT(ConfigMixin):

    def __init__(self) -> None:

        self.general: Dict[str, Any] = {
                'device': 'cpu',
                'disable_component_lock': False,
                'seed': 13,
                'description': "No description", # Should provide a basic description of this MetaCAT model
                'category_name': None, # What category is this meta_cat model predicting/training
                'category_value2id': {}, # Map from category values to ID, if empty it will be autocalculated during training
                'vocab_size': None, # Will be set automatically if the tokenizer is provided during meta_cat init
                'lowercase': True, # If true all input text will be lowercased
                'cntx_left': 15, # Number of tokens to take from the left of the concept
                'cntx_right': 10, # Number of tokens to take from the right of the concept
                'replace_center': None, # If set the center (concept) will be replaced with this string
                'batch_size_eval': 5000, # Number of annotations to be meta-annotated at once in eval
                'annotate_overlapping': False, # If set meta_anns will be calcualted for doc._.ents, otherwise for doc.ents
                'tokenizer_name': 'bbpe', # Tokenizer name used with of MetaCAT
                # This is a dangerous option, if not sure ALWAYS set to False. If set, it will try to share the pre-calculated
                #context tokens between MetaCAT models when serving. It will ignore differences in tokenizer and context size,
                #so you need to be sure that the models for which this is turned on have the same tokenizer and context size, during
                #a deployment.
                'save_and_reuse_tokens': False,
                'pipe_batch_size_in_chars': 20000000, # How many characters are piped at once into the meta_cat class
                }
        self.model: Dict[str, Any] = {
                'model_name': 'lstm',
                'num_layers': 2,
                'input_size': 300,
                'hidden_size': 300,
                'dropout': 0.5,
                'num_directions': 2, # 2 - bidirectional model, 1 - unidirectional
                'nclasses': 2, # Number of classes that this model will output
                'padding_idx': -1,
                'emb_grad': True, # If True the embeddings will also be trained
                'ignore_cpos': False, # If set to True center positions will be ignored when calculating represenation
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
                'cui_filter': None, # If set only this CUIs will be used for training
                'auto_save_model': True, # Should do model be saved during training for best results
                'last_train_on': None, # When was the last training run
                'metric': {'base': 'weighted avg', 'score': 'f1-score'}, # What metric should be used for choosing the best model
                }
