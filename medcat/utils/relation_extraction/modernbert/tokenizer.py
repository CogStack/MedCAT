import os
from transformers import PretrainedConfig
from transformers import ModernBertConfig
from transformers import PreTrainedTokenizerFast
import logging

from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper
from medcat.utils.relation_extraction.models import Base_RelationExtraction
from medcat.utils.relation_extraction.modernbert.model import ModernBertModel_RelationExtraction


logger = logging.getLogger(__name__)


class TokenizerWrapperModernBERT(BaseTokenizerWrapper):
    ''' Wrapper around a huggingface ModernBERT tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.PreTrainedTokenizerFast`):
            A huggingface Fast tokenizer.
    '''
    name = 'modern-bert-tokenizer'
    pretrained_model_name_or_path = "answerdotai/ModernBERT-base"

    def config_from_pretrained(self) -> PretrainedConfig:
        return ModernBertConfig.from_pretrained(self.pretrained_model_name_or_path)

    def config_from_json_file(self, file_path: str) -> PretrainedConfig:
        return ModernBertConfig.from_json_file(file_path)

    def model_from_pretrained(self, relcat_config: ConfigRelCAT, model_config: PretrainedConfig,
            pretrained_model_name_or_path: str = 'default') -> Base_RelationExtraction:
        if pretrained_model_name_or_path == 'default':
            pretrained_model_name_or_path = self.pretrained_model_name_or_path
        return ModernBertModel_RelationExtraction(pretrained_model_name_or_path, relcat_config, model_config)

    @classmethod
    def load(cls, dir_path, **kwargs):
        tokenizer = cls()
        path = os.path.join(dir_path, cls.name)
        tokenizer.hf_tokenizers = PreTrainedTokenizerFast.from_pretrained(
            path, **kwargs)

        return tokenizer
