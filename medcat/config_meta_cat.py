import logging
from typing import Dict, Any, List
from collections.abc import Container
from medcat.config import MixingConfig, BaseModel, Optional


logger = logging.getLogger(__name__)


class General(MixingConfig, BaseModel):
    """The General part of the MetaCAT config"""
    device: str = 'cpu'
    """
    Device to used by the module to perform predicting/training.

    Reference: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
    """
    disable_component_lock: bool = False
    """ Whether to use the MetaCAT component lock.

    If set to False (the default), a component lock is used that forces usage only on one thread at a time.

    If set to True, the component lock is not used."""
    seed: int = 13
    """The seed for random number generation.

    NOTE: If used along RelCAT or additional NER, only one of the seeds will take effect
    NB! For these changes to take effect, the pipe would need to be recreated."""
    description: str = "No description"
    """Should provide a basic description of this MetaCAT model"""
    category_name: Optional[str] = None
    """What category is this meta_cat model predicting/training.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    alternative_category_names: List = []
    """List that stores the variations of possible category names
    Example: For Experiencer, the alternate name is Subject
    alternative_category_names: ['Experiencer','Subject']

    In the case that one specified in self.general.category_name parameter does not match the data, this ensures no error is raised and it is automatically mapped
    """
    category_value2id: Dict = {}
    """Map from category values to ID, if empty it will be autocalculated during training"""
    alternative_class_names: List[List] = [[]]
    """List of lists that stores the variations of possible class names for each class mentioned in self.general.category_value2id

    Example: For Presence task, the class names vary across NHS sites.
    To accommodate for this, alternative_class_names is populated as: [["Hypothetical (N/A)","Hypothetical"],["Not present (False)","False"],["Present (True)","True"]]
    Each sub list contains the possible variations of the given class.
    """
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
    """If set meta_anns will be calculated for doc._.ents, otherwise for doc.ents"""
    tokenizer_name: str = 'bbpe'
    """
    Tokenizer name used with MetaCAT.

    Choose from:
        - 'bbpe': Byte Pair Encoding Tokenizer
        - 'bert-tokenizer': BERT Tokenizer

    NB! For these changes to take effect, the pipe would need to be recreated.
    """
    save_and_reuse_tokens: bool = False
    """This is a dangerous option, if not sure ALWAYS set to False. If set, it will try to share the pre-calculated
    context tokens between MetaCAT models when serving. It will ignore differences in tokenizer and context size,
    so you need to be sure that the models for which this is turned on have the same tokenizer and context size, during
    a deployment."""
    pipe_batch_size_in_chars: int = 20000000
    """How many characters are piped at once into the meta_cat class"""
    span_group: Optional[str] = None
    """If set, the spacy span group that the metacat model will assign annotations.
    Otherwise defaults to doc._.ents or doc.ents per the annotate_overlapping settings"""

    def get_applicable_category_name(self, available_names: Container[str]) -> Optional[str]:
        if self.category_name in available_names:
            return self.category_name
        matches = [cat for cat in self.alternative_category_names if cat in available_names]
        if len(matches) > 0:
            logger.info("The category name provided in the config - '%s' is not present in the data. "
                        "However, the corresponding name - '%s' from the category_name_mapping has been found. "
                        "Updating the category name...", self.category_name, *matches)
            self.category_name = matches[0]
            return self.category_name
        return None

    class Config:
        extra = 'allow'
        validate_assignment = True


class Model(MixingConfig, BaseModel):
    """The model part of the metaCAT config"""
    model_name: str = 'lstm'
    """
    Model to be used for training or predicting.

    Choose from:
        - 'bert'
        - 'lstm'

    Note:
        When changing the model, make sure to change the tokenizer accordingly.
        NB! For these changes to take effect, the pipe would need to be recreated.
    """
    model_variant: str = 'bert-base-uncased'
    """
    Applicable only when using BERT:

    Specifies the model variant to be used.

    NB! For these changes to take effect, the pipe would need to be recreated.
    """
    model_freeze_layers: bool = True
    """
    Applicable only when using BERT:

    Determines the training approach for BERT.

    - If True: BERT layers are frozen and only the fully connected (FC) layer(s) on top are trained.
    - If False: Parameter-efficient fine-tuning will be applied using Low-Rank Adaptation (LoRA).

    NB! For these changes to take effect, the pipe would need to be recreated.
    """
    load_bert_pretrained_weights: bool = False
    """
    Applicable only when using BERT:
    Determines if the pretrained weights for BERT are loaded
    This should be True if you don't plan on using the model pack weights"""

    num_layers: int = 2
    """Number of layers in the model (both LSTM and BERT)

    NB! For these changes to take effect, the pipe would need to be recreated."""
    input_size: int = 300
    """
    Specifies the size of the embedding layer.

    Applicable only for LSTM model and ignored for BERT as BERT's embedding size is predefined.

    NB! For these changes to take effect, the pipe would need to be recreated.
    """
    hidden_size: int = 300
    """Number of neurons in the hidden layer.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    dropout: float = 0.5
    """The dropout for the model.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    phase_number: int = 0
    """Indicates whether two phase learning is to be used for training.

    1: Phase 1 - Train model on undersampled data

    2: Phase 2 - Continue training on full data

    0: None - 2 phase learning is not performed

    Paper reference - https://ieeexplore.ieee.org/document/7533053"""
    category_undersample: str = ''
    """When using 2 phase learning, this category is used to undersample the data
    The number of samples in the category sets the upper limit for all categories"""

    model_architecture_config: Dict = {'fc2': True, 'fc3': False,'lr_scheduler': True}
    """Specifies the architecture for BERT model.

    If fc2 is set to True, then the 2nd fully connected layer is used

    If fc2 is True and fc3 is set to True, then the 3rd fully connected layer is used

    If lr_scheduler is set to True, then the learning rate scheduler is used with the optimizer

    NB! For these changes to take effect, the pipe would need to be recreated.
    """
    num_directions: int = 2
    """Applicable only for LSTM:

    2 - bidirectional model, 1 - unidirectional

    NB! For these changes to take effect, the pipe would need to be recreated."""
    nclasses: int = 2
    """Number of classes that this model will output.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    padding_idx: int = -1
    """The padding index.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    emb_grad: bool = True
    """Applicable only for LSTM:

    If True, the embeddings will also be trained.

    NB! For these changes to take effect, the pipe would need to be recreated."""
    ignore_cpos: bool = False
    """If set to True center positions will be ignored when calculating representation"""

    class Config:
        extra = 'allow'
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
    compute_class_weights: bool = False
    """If true and class weights not provided, the class weights will be calculated based on the data"""
    score_average: str = 'weighted'
    """What to use for averaging F1/P/R across labels"""
    prerequisites: dict = {}
    cui_filter: Optional[Any] = None
    """If set only this CUIs will be used for training"""
    auto_save_model: bool = True
    """Should do model be saved during training for best results"""
    last_train_on: Optional[float] = None
    """When was the last training run"""
    metric: Dict[str, str] = {'base': 'weighted avg', 'score': 'f1-score'}
    """What metric should be used for choosing the best model"""
    loss_funct: str = 'cross_entropy'
    """Loss function for the model.

    Choose from:
        - 'cross_entropy'
        - 'focal_loss'
    """
    gamma: int = 2
    """Focal Loss hyperparameter - determines importance the loss gives to hard-to-classify examples"""

    class Config:
        extra = 'allow'
        validate_assignment = True


class ConfigMetaCAT(MixingConfig, BaseModel):
    """The MetaCAT part of the config"""
    general: General = General()
    model: Model = Model()
    train: Train = Train()

    class Config:
        extra = 'allow'
        validate_assignment = True
