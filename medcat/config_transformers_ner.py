from typing import Dict, Any
from medcat.config import ConfigMixin
from transformers import TrainingArguments


class ConfigTransformersNER(ConfigMixin):

    def __init__(self) -> None:

        self.general: Dict[str, Any] = {
                'model_name': 'roberta-base', # Can be path also
                'seed': 13,
                'description': "No description", # Should provide a basic description of this MetaCAT model
                'pipe_batch_size_in_chars': 20000000, # How many characters are piped at once into the meta_cat class
                }

        self.train = TrainingArguments(
                output_dir='./results',
                logging_dir='./logs',            # directory for storing logs
                num_train_epochs=10,              # total number of training epochs
                per_device_train_batch_size=1,  # batch size per device during training
                per_device_eval_batch_size=1,   # batch size for evaluation
                weight_decay=0.14,               # strength of weight decay
                warmup_ratio=0.01,
                learning_rate= 4.47e-05, # Should be smaller when finetuning an existing deid model
                eval_accumulation_steps=1,
                gradient_accumulation_steps=4, # We want to get to bs=4
                do_eval=True,
                evaluation_strategy='epoch',
                save_strategy='epoch',
                metric_for_best_model='eval_recall', # Can be changed if our preference is not recall but something else
                load_best_model_at_end=True,
                remove_unused_columns=False)

        # Add a couple of more things to training_arguments
        self.train.test_size = 0.1
        self.train.shuffle_data = True
        self.train.last_train_on = None

        """
        self.train: Dict[str, Any] = {
                'test_size': 0.1,
                'shuffle_data': True, # Used only during training, if set the dataset will be shuffled before train/test split
                'score_average': 'weighted', # What to use for averaging F1/P/R across labels
                'meta-requirements': {},
                'cui_filter': None, # If set only this CUIs will be used for training
                'last_train_on': None, # When was the last training run
                'metric': {'base': 'weighted avg', 'score': 'f1-score'}, # What metric should be used for choosing the best model
                }
        """


