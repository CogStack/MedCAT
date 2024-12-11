import logging
from typing import Dict, Any, List
from medcat.config import MixingConfig, BaseModel, Optional


class General(MixingConfig, BaseModel):
    """The General part of the RelCAT config"""
    device: str = "cpu"
    """The device to use (CPU or GPU).

    NB! For these changes to take effect, the pipe would need to be recreated."""
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

    mct_export_max_non_rel_sample_size:int = 200
    """Limit the number of 'Other' samples selected for training/test. This is applied per encountered medcat project, sample_size/num_projects. """
    mct_export_create_addl_rels: bool = False
    """When processing relations from a MedCAT export, relations labeled as 'Other' are created from all the annotations pairs available"""

    tokenizer_name: str = "bert"
    """The name of the tokenizer user.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    model_name: str = "bert-base-uncased"
    """The name of the model used.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    log_level: int = logging.INFO
    """The log level for RelCAT.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    max_seq_length: int = 512
    """The maximum sequence length.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    tokenizer_special_tokens: bool = False
    """Tokenizer.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    annotation_schema_tag_ids: List = []
    """If a foreign non-MCAT trainer dataset is used, you can insert your own Rel entity token delimiters into the tokenizer, \
    copy those token IDs here, and also resize your tokenizer embeddings and adjust the hidden_size of the model, this will depend on the number of tokens you introduce"""
    labels2idx: Dict = {}
    idx2labels: Dict = {}
    pin_memory: bool = True
    seed: int = 13
    """The seed for random number generation.

    NOTE: If used along MetaCAT or additional NER, only one of the seeds will take effect
    NB! For these changes to take effect, the pipe would need to be recreated."""
    task: str = "train"
    """The task for RelCAT.

    NB! For these changes to take effect, the pipe would need to be recreated."""


class Model(MixingConfig, BaseModel):
    """The model part of the RelCAT config"""
    input_size: int = 300
    hidden_size: int = 768
    """The hidden size.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    hidden_layers: int = 3
    """ hidden_size * 5, 5 being the number of tokens, default (s1,s2,e1,e2+CLS).

    NB! For these changes to take effect, the pipe would need to be recreated."""
    model_size: int = 5120
    """The size of the model.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    dropout: float = 0.2
    num_directions: int = 2
    """2 - bidirectional model, 1 - unidirectional"""

    padding_idx: int = -1
    emb_grad: bool = True
    """If True the embeddings will also be trained"""
    ignore_cpos: bool = False
    """If set to True center positions will be ignored when calculating representation"""

    class Config:
        extra = 'allow'
        validate_assignment = True


class Train(MixingConfig, BaseModel):
    """The train part of the RelCAT config"""
    nclasses: int = 2
    """Number of classes that this model will output"""
    batch_size: int = 25
    nepochs: int = 1
    lr: float = 1e-4
    adam_epsilon: float = 1e-4
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
        extra = 'allow'
        validate_assignment = True


class ConfigRelCAT(MixingConfig, BaseModel):
    """The RelCAT part of the config"""
    general: General = General()
    model: Model = Model()
    train: Train = Train()

    class Config:
        extra = 'allow'
        validate_assignment = True
