from medcat.config import MixingConfig, BaseModel, Optional, Extra


class General(MixingConfig, BaseModel):
    """The general part of the Transformers NER config"""
    pipe_batch_size_in_chars: int = 20000000
    """How many characters are piped at once into the meta_cat class"""
    ner_aggregation_strategy: str = 'simple'
    """Agg strategy for HF pipeline for NER"""
    chunking_overlap_window: Optional[int] = 5
    """Size of the overlap window used for chunking"""
    test_size: float = 0.2
    last_train_on: Optional[int] = None
    verbose_metrics: bool = False

    class Config:
        extra = Extra.allow
        validate_assignment = True


class PreLoad(MixingConfig, BaseModel):
    """The parts of config that only take effect when setting before loading a NER model.

    Changes to the parameters listed here will generally only be effective if done
    before a model is initialised or loaded.
    """
    name: str = 'deid'
    model_name: str = 'roberta-base'
    """Can be path also"""
    seed: int = 13
    """NOTE: If used along RelCAT or RelCAT, only one of the seeds will take effect"""
    description: str = "No description"
    """Should provide a basic description of this MetaCAT model"""


class ConfigTransformersNER(MixingConfig, BaseModel):
    """The transformer NER config"""
    general: General = General()
    pre_load: PreLoad = PreLoad()

    class Config:
        extra = Extra.allow
        validate_assignment = True
