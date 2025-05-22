import os
import logging
from typing import Any, Dict, List, Tuple, Union, cast
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

    limit_samples_per_class: int = -1
    """Number of samples per class, this limit is applied for train samples, so if train samples are 100 then test would be 20."""
    addl_rels_max_sample_size:int = 200
    """Limit the number of 'Other' samples selected for training/test. This is applied per encountered medcat project, sample_size/num_projects. """
    create_addl_rels: bool = False
    """When processing relations from a MedCAT export/docs, relations labeled as 'Other' are created from all the annotations pairs available"""
    create_addl_rels_by_type: bool = False
    """When creating the 'Other' relation class, actually split this class into subclasses based on concept types"""

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
    annotation_schema_tag_ids: List = [30522, 30523, 30524, 30525]
    """If a foreign non-MCAT trainer dataset is used, you can insert your own Rel entity token delimiters into the tokenizer, \
    copy those token IDs here, and also resize your tokenizer embeddings and adjust the hidden_size of the model, this will depend on the number of tokens you introduce
    for example: 30522 - [s1], 30523 - [e1], 30524 - [s2], 30525 - [e2], 30526 - [BLANK], 30527 - [ENT1], 30528 - [ENT2], 30529 - [/ENT1], 30530 - [/ENT2]
    Please note that the tokenizer special tokens are supposed to be in pairs of two for example [s1] and [e1], [s2] and [e2], the [BLANK] is just an example placeholder token
    If you have more than four tokens here then you need to make sure they are present in the text, 
    otherwise the pipeline will throw an error in the get_annotation_schema_tag() function.
    """

    tokenizer_relation_annotation_special_tokens_tags: List[str] = ["[s1]", "[e1]", "[s2]", "[e2]"]

    tokenizer_other_special_tokens: Dict[str, str] = {"pad_token": "[PAD]"}
    """
    The special tokens used by the tokenizer. The {PAD} is for Lllama tokenizer."""

    labels2idx: Dict[str, int] = {}
    idx2labels: Dict[int, str] = {}

    pin_memory: bool = True
    """If True the data loader will copy the tensors to the GPU pinned memory"""

    seed: int = 13
    """The seed for random number generation.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    task: str = "train"
    """The task for RelCAT."""

    language: str = "en"
    """Used for Spacy lang setting"""

    @classmethod
    def convert_keys_to_int(cls, value):
        if isinstance(value, dict):
            return {int(k): v for k, v in value.items()}
        return value

    def __setattr__(self, key: str, value: Any):
        if key == "idx2labels" and isinstance(value, dict):
            value = self.convert_keys_to_int(value)  # Ensure conversion
        super().__setattr__(key, value)


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

    freeze_layers: bool = True
    """If we update the weights during training"""

    padding_idx: int = -1
    emb_grad: bool = True
    """If True the embeddings will also be trained"""
    ignore_cpos: bool = False
    """If set to True center positions will be ignored when calculating representation"""

    llama_use_pooled_output: bool = False
    """If set to True, used only in Llama model, it will add the extra tensor formed from selecting the max of the last hidden layer"""

    class Config:
        extra = 'allow'
        validate_assignment = True


class Train(MixingConfig, BaseModel):
    """The train part of the RelCAT config"""
    nclasses: int = 2
    """Number of classes that this model will output"""
    batch_size: int = 25
    """batch size"""
    nepochs: int = 1
    """Epochs"""
    lr: float = 1e-4
    """Learning rate"""
    stratified_batching: bool = False
    """Train the model with stratified batching"""
    batching_samples_per_class: list = []
    """Number of samples per class in each batch
    example for batch size 64: [6,6,6,8,8,8,6,8,8]"""
    batching_minority_limit: Union[List[int], int] = 0
    """Maximum number of samples the minority class can have.
    Since the minority class elements need to be repeated, this is used to facilitate that
    example: batching_samples_per_class - [6,6,6,8,8,8,6,8,8]
             batching_minority_limit - 6"""
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_weight_decay: float = 0
    adam_epsilon: float = 1e-8
    test_size: float = 0.2
    gradient_acc_steps: int = 1
    multistep_milestones: List[int] = [
        2, 4, 6, 8, 12, 15, 18, 20, 22, 24, 26, 30]
    multistep_lr_gamma: float = 0.8
    max_grad_norm: float = 1.0
    shuffle_data: bool = True
    """Used only during training, if set the dataset will be shuffled before train/test split"""
    class_weights: Union[List[float], None] = None
    enable_class_weights: bool = False
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

    @classmethod
    def load(cls, load_path: str = "./") -> "ConfigRelCAT":
        """Load the config from a file.

        Args:
            load_path (str): Path to RelCAT config. Defaults to "./".

        Returns:
            ConfigRelCAT: The loaded config.
        """
        config = cls()
        if os.path.exists(load_path):
            if "config.json" not in load_path:
                load_path = os.path.join(load_path, "config.json")
            config = cast(ConfigRelCAT, super().load(load_path))
            logging.info("Loaded config.json")

        return config
