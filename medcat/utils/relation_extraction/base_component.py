from abc import ABC, abstractmethod
import logging

from transformers import PretrainedConfig

from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper
from medcat.utils.relation_extraction.models import Base_RelationExtraction


logger = logging.getLogger(__name__)


class BaseComponent(ABC):

    @property
    @abstractmethod
    def tokenizer(self) -> BaseTokenizerWrapper:
        pass

    @abstractmethod
    def config_from_pretrained(self) -> PretrainedConfig:
        pass # perhaps some doc string

    @abstractmethod
    def config_from_json_file(self, file_path: str) -> PretrainedConfig:
        pass # perhaps some doc string

    @abstractmethod
    def model_from_pretrained(self, relcat_config: ConfigRelCAT, model_config: PretrainedConfig,
            pretrained_model_name_or_path: str = 'default') -> Base_RelationExtraction:
        pass # perhaps some doc string


def load_base_component(tokenizer_path: str, config: ConfigRelCAT) -> BaseComponent:
    if "modern-bert-tokenizer" in config.general.tokenizer_name:
        from medcat.utils.relation_extraction.modernbert.component import ModernBertComponent
        return ModernBertComponent(tokenizer_path, config)
    elif "bert" in config.general.tokenizer_name:
        from medcat.utils.relation_extraction.bert.component import BertComponent
        return BertComponent(tokenizer_path, config)
    elif "llama" in config.general.tokenizer_name:
        from medcat.utils.relation_extraction.llama.component import LlamaComponent
        return LlamaComponent(tokenizer_path, config)
    raise ValueError(f"Could not find matching base component for {config.general.tokenizer_name}")
