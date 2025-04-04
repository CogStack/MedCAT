import os

from transformers import PretrainedConfig
from transformers.models.llama import LlamaConfig

from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.base_component import BaseComponent
from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper, load_default_tokenizer
from medcat.utils.relation_extraction.models import Base_RelationExtraction
from medcat.utils.relation_extraction.llama.tokenizer import TokenizerWrapperLlama
from medcat.utils.relation_extraction.llama.model import LlamaModel_RelationExtraction


class LlamaComponent(BaseComponent):
    pretrained_model_name_or_path = "meta-llama/Llama-3.1-8B"

    def __init__(self, tokenizer_path: str, config: ConfigRelCAT):
        if os.path.exists(tokenizer_path):
            self._tokenizer = TokenizerWrapperLlama.load(tokenizer_path)
        else:
            self._tokenizer = load_default_tokenizer(tokenizer_path, config)

    @property
    def tokenizer(self) -> BaseTokenizerWrapper:
        return self._tokenizer

    def config_from_pretrained(self) -> PretrainedConfig:
        pass # perhaps some doc string

    def config_from_json_file(self, file_path: str) -> PretrainedConfig:
        return LlamaConfig.from_json_file(file_path)

    def model_from_pretrained(self, relcat_config: ConfigRelCAT, model_config: PretrainedConfig,
            pretrained_model_name_or_path: str = 'default') -> Base_RelationExtraction:
        if pretrained_model_name_or_path == 'default':
            pretrained_model_name_or_path = self.pretrained_model_name_or_path
        return LlamaModel_RelationExtraction(pretrained_model_name_or_path, relcat_config, model_config)
