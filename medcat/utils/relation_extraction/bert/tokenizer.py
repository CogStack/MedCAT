import os
from abc import abstractmethod
from transformers import PretrainedConfig
from transformers import BertConfig
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import logging

from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper
from medcat.utils.relation_extraction.models import Base_RelationExtraction
from medcat.utils.relation_extraction.bert.model import BertModel_RelationExtraction


logger = logging.getLogger(__name__)


class TokenizerWrapperBERT(BaseTokenizerWrapper):
    ''' Wrapper around a huggingface BERT tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.models.bert.tokenization_bert_fast.PreTrainedTokenizerFast`):
            A huggingface Fast BERT.
    '''
    name = 'bert-tokenizer'
    pretrained_model_name_or_path = "bert-base-uncased"

    @abstractmethod
    def config_from_pretrained(self) -> PretrainedConfig:
        return BertConfig.from_pretrained(self.pretrained_model_name_or_path)

    @abstractmethod
    def config_from_json_file(self, file_path: str) -> PretrainedConfig:
        return BertConfig.from_json_file(file_path)

    @abstractmethod
    def model_from_pretrained(self, relcat_config: ConfigRelCAT, model_config: PretrainedConfig,
            pretrained_model_name_or_path: str = 'default') -> Base_RelationExtraction:
        if pretrained_model_name_or_path == 'default':
            pretrained_model_name_or_path = self.pretrained_model_name_or_path
        return BertModel_RelationExtraction(pretrained_model_name_or_path, relcat_config, model_config)

    @classmethod
    def load(cls, dir_path, **kwargs):
        tokenizer = cls()
        path = os.path.join(dir_path, cls.name)
        tokenizer.hf_tokenizers = BertTokenizerFast.from_pretrained(
            path, **kwargs)

        return tokenizer
