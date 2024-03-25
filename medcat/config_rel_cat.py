import logging
from typing import Dict, Any, List
from medcat.config import MixingConfig, BaseModel, Optional, Extra


class General(MixingConfig, BaseModel):
    """The General part of the RelCAT config"""
    device: str = "cpu"
    relation_type_filter_pairs: List = []
    """Map from category values to ID, if empty it will be autocalculated during training"""
    vocab_size: Optional[int] = None
    lowercase: bool = True
    """If true all input text will be lowercased"""
    cntx_left: int = 15
    """Number of tokens to take from the left of the concept"""
    cntx_right: int = 15
    """Number of tokens to take from the right of the concept"""
    window_size: int = 300
    """Max acceptable dinstance between entities (in characters), care when using this as it can produce sentences that are over 512 tokens (limit is given by tokenizer)"""
    tokenizer_name: str = "BERT_tokenizer_relation_extraction"
    model_name: str = "bert-base-uncased"
    log_level: int = logging.INFO
    max_seq_length: int = 512
    tokenizer_special_tokens: bool = False
    annotation_schema_tag_ids: List = []
    """If a foreign non-MCAT trainer dataset is used, you can insert your own Rel entity token delimiters into the tokenizer, \
    copy those token IDs here, and also resize your tokenizer embeddings and adjust the hidden_size of the model, this will depend on the number of tokens you introduce"""
    labels2idx: Dict = {}
    idx2labels: Dict = {}
    pin_memory: bool = True
    seed: int = 13
    task: str = "train"


class Model(MixingConfig, BaseModel):
    """The model part of the RelCAT config"""
    input_size: int = 300
    hidden_size: int = 300
    dropout: float = 0.1
    num_directions: int = 2
    """2 - bidirectional model, 1 - unidirectional"""

    padding_idx: int = -1
    emb_grad: bool = True
    """If True the embeddings will also be trained"""
    ignore_cpos: bool = False
    """If set to True center positions will be ignored when calculating represenation"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class Train(MixingConfig, BaseModel):
    """The train part of the RelCAT config"""
    nclasses: int = 2
    """Number of classes that this model will output"""
    batch_size: int = 25
    nepochs: int = 2
    lr: float = 1e-5
    adam_epsilon: float = 1e-8
    test_size: float = 0.2
    gradient_acc_steps: int = 1
    multistep_milestones: List[int] = [
        2, 4, 6, 8, 12, 15, 18, 20, 22, 24, 26, 30]
    multistep_lr_gamma: float = 0.8
    max_grad_norm: float = 1.0
    shuffle_data: bool = True
    """Used only during training, if set the dataset will be shuffled before train/test split"""
    class_weights: Optional[Any] = None
    score_average: str = "weighted"
    """What to use for averaging F1/P/R across labels"""
    auto_save_model: bool = True
    """Should the model be saved during training for best results"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class ConfigRelCAT(MixingConfig, BaseModel):
    """The RelCAT part of the config"""
    general: General = General()
    model: Model = Model()
    train: Train = Train()

    class Config:
        extra = Extra.allow
        validate_assignment = True
