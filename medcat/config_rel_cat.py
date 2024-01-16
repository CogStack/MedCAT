import logging
from typing import Dict, Any
from medcat.config import MixingConfig


class ConfigRelCAT(MixingConfig):

    def __init__(self) -> None:

        self.general: Dict[str, Any] = {
                'device': 'cuda', # possible values: 'cuda', 'cpu'
                'relation_type_filter_pairs': [], # these are the pairs of relations that are to be predicted , form of : [("Disease", "Symptom"), ("entity1_type", "entity2_type") ...]
                'vocab_size': None, # Will be set automatically if the tokenizer is provided during rel_cat init/load
                'lowercase': True, # If true all input text will be lowercased
                'ent_context_left': 2, # Number of entities to take from the left of the concept
                'ent_context_right': 2, # Number of entities to take from the right of the concept
                'window_size': 300, # max acceptable dinstance between entities (in characters), care when using this as it can produce sentences that are over 512 tokens (limit is given by tokenizer)
                'tokenizer_name': 'BERT_tokenizer_relation_extraction', # Tokenizer name used, "BERT_tokenizer_relation_extraction" default
                'model_name': 'bert-base-uncased', # e.g "dmis-lab/biobert-base-cased-v1.2", "bert-large-uncased", "bert-base-uncased", "emilyalsentzer/Bio_ClinicalBERT"
                'log_level': logging.INFO,
                'max_seq_length': 512,
                'padding_idx': -1,
                'task': 'train',
                'tokenizer_special_tokens': False,
                'annotation_schema_tag_ids': [30522, 30523, 30524, 30525], # s1,s2,e1,e2 token id markers
                'labels2idx': {},
                'idx2labels': {},
                'pin_memory': True,
                "seed": 13
        }

        self.model: Dict[str, Any] = {
                'input_size': 300,
                'hidden_size': 512, # model_size
                'dropout': 0.1,
                'nclasses': 2, # Number of classes that this model will output
                }

        self.train: Dict[str, Any] = {
                'batch_size': 25,
                'nepochs': 2,
                'lr': 0.00001,
                'adam_epsilon': 1e-8,
                'lr': 1e-5,
                'test_size': 0.2,
                'gradient_acc_steps': 1,
                'multistep_milestones': [2,4,6,8,12,15,18,20,22,24,26,30],
                'multistep_lr_gamma': 0.8,
                'max_grad_norm': 1.0,
                'shuffle_data': True, # Used only during training, if set the dataset will be shuffled before train/test split
                'class_weights': None,
                'score_average': 'weighted', # What to use for averaging F1/P/R across labels
                'prerequisites': {},
                'auto_save_model': True, # Should do model be saved during training for best results
                }
