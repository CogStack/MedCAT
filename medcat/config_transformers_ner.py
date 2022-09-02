from typing import Dict, Any
from medcat.config import ConfigMixin

from medcat.idconfig import MixingConfig, BaseModel, Optional, Extra

class _ConfigTransformersNER(ConfigMixin):

    def __init__(self) -> None:

        self.general: Dict[str, Any] = {
                'name': 'deid',
                'model_name': 'roberta-base', # Can be path also
                'seed': 13,
                'description': "No description", # Should provide a basic description of this MetaCAT model
                'pipe_batch_size_in_chars': 20000000, # How many characters are piped at once into the meta_cat class
                'ner_aggregation_strategy': 'simple', # Agg strategy for HF pipeline for NER
                'test_size': 0.2,
                'last_train_on': None,
                'verbose_metrics': False,
                }

class General(MixingConfig, BaseModel):
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

class ConfigTransformersNER(MixingConfig, BaseModel):
    general: General = General()


    class Config:
        extra = Extra.allow
