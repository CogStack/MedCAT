import os
from typing import Optional

from transformers import PretrainedConfig, BertConfig

from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.base_component import BaseComponent
from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper, load_default_tokenizer
from medcat.utils.relation_extraction.models import Base_RelationExtraction
from medcat.utils.relation_extraction.bert.tokenizer import TokenizerWrapperBERT
from medcat.utils.relation_extraction.bert.model import BertModel_RelationExtraction


class BertComponent(BaseComponent):
    pretrained_model_name_or_path = "bert-base-uncased"

    def __init__(self, tokenizer_path: str, config: ConfigRelCAT,
                 tokenizer: Optional[BaseTokenizerWrapper] = None):
        if tokenizer is not None:
            self._tokenizer = tokenizer
        elif os.path.exists(tokenizer_path):
            self._tokenizer = TokenizerWrapperBERT.load(tokenizer_path)
        else:
            self._tokenizer = load_default_tokenizer(tokenizer_path, config)

    @property
    def tokenizer(self) -> BaseTokenizerWrapper:
        return self._tokenizer

    def config_from_pretrained(self) -> PretrainedConfig:
        return BertConfig.from_pretrained(self.pretrained_model_name_or_path)

    def config_from_json_file(self, file_path: str) -> PretrainedConfig:
        return BertConfig.from_json_file(file_path)

    def model_from_pretrained(self, relcat_config: ConfigRelCAT, model_config: PretrainedConfig,
            pretrained_model_name_or_path: str = 'default') -> Base_RelationExtraction:
        if pretrained_model_name_or_path == 'default':
            pretrained_model_name_or_path = self.pretrained_model_name_or_path
        return BertModel_RelationExtraction(pretrained_model_name_or_path, relcat_config, model_config)
