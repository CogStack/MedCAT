from typing import Dict, Any

from medcat.config import MixingConfig, BaseModel, Optional, Extra


class General(MixingConfig, BaseModel):
    """The General part of the MetaCAT config"""
    device: str = 'cpu'
    disable_component_lock: bool = False
    seed: int = 13
    description: str = "No description"
    """Should provide a basic description of this MetaCAT model"""
    category_name: Optional[str] = None
    """What category is this meta_cat model predicting/training"""
    category_value2id: Dict = {}
    """Map from category values to ID, if empty it will be autocalculated during training"""
    vocab_size: Optional[int] = None
    """Will be set automatically if the tokenizer is provided during meta_cat init"""
    lowercase: bool = True
    """If true all input text will be lowercased"""
    cntx_left: int = 15
    """Number of tokens to take from the left of the concept"""
    cntx_right: int = 10
    """Number of tokens to take from the right of the concept"""
    replace_center: Optional[Any] = None
    """If set the center (concept) will be replaced with this string"""
    batch_size_eval: int = 5000
    """Number of annotations to be meta-annotated at once in eval"""
    annotate_overlapping: bool = False
    """If set meta_anns will be calcualted for doc._.ents, otherwise for doc.ents"""
    tokenizer_name: str = 'bbpe'
    """Tokenizer name used with of MetaCAT"""
    save_and_reuse_tokens: bool = False
    """This is a dangerous option, if not sure ALWAYS set to False. If set, it will try to share the pre-calculated
    context tokens between MetaCAT models when serving. It will ignore differences in tokenizer and context size,
    so you need to be sure that the models for which this is turned on have the same tokenizer and context size, during
    a deployment."""
    pipe_batch_size_in_chars: int = 20000000
    """How many characters are piped at once into the meta_cat class"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class Model(MixingConfig, BaseModel):
    """The model part of the metaCAT config"""
    model_name: str = 'lstm'
    num_layers: int = 2
    input_size: int = 300
    hidden_size: int = 300
    dropout: float = 0.5
    num_directions: int = 2
    """2 - bidirectional model, 1 - unidirectional"""
    nclasses: int = 2
    """Number of classes that this model will output"""
    padding_idx: int = -1
    emb_grad: bool = True
    """If True the embeddings will also be trained"""
    ignore_cpos: bool = False
    """If set to True center positions will be ignored when calculating represenation"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class Train(MixingConfig, BaseModel):
    """The train part of the metaCAT config"""
    batch_size: int = 100
    nepochs: int = 50
    lr: float = 0.001
    test_size: float = 0.1
    shuffle_data: bool = True
    """Used only during training, if set the dataset will be shuffled before train/test split"""
    class_weights: Optional[Any] = None
    score_average: str = 'weighted'
    """What to use for averaging F1/P/R across labels"""
    prerequisites: dict = {}
    cui_filter: Optional[Any] = None
    """If set only this CUIs will be used for training"""
    auto_save_model: bool = True
    """Should do model be saved during training for best results"""
    last_train_on: Optional[int] = None
    """When was the last training run"""
    metric: Dict[str, str] = {'base': 'weighted avg', 'score': 'f1-score'}
    """What metric should be used for choosing the best model"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class ConfigMetaCAT(MixingConfig, BaseModel):
    """The MetaCAT part of the config"""
    general: General = General()
    model: Model = Model()
    train: Train = Train()

    class Config:
        extra = Extra.allow
        validate_assignment = True
