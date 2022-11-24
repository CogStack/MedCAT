from medcat.config import MixingConfig, BaseModel, Optional, Extra


class General(MixingConfig, BaseModel):
    """The general part of the Transformers NER config"""
    name: str = 'deid'
    model_name: str = 'roberta-base'
    """Can be path also"""
    seed: int = 13
    description: str = "No description"
    """Should provide a basic description of this MetaCAT model"""
    pipe_batch_size_in_chars: int = 20000000
    """How many characters are piped at once into the meta_cat class"""
    ner_aggregation_strategy: str = 'simple'
    """Agg strategy for HF pipeline for NER"""
    test_size: float = 0.2
    last_train_on: Optional[int] = None
    verbose_metrics: bool = False

    class Config:
        extra = Extra.allow
        validate_assignment = True


class ConfigTransformersNER(MixingConfig, BaseModel):
    """The transformer NER config"""
    general: General = General()

    class Config:
        extra = Extra.allow
        validate_assignment = True
