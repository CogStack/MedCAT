import os
from abc import ABC, abstractmethod
from typing import Optional
from transformers import PretrainedConfig
from transformers import BertConfig, ModernBertConfig
from transformers import AutoTokenizer
from transformers.models.llama import LlamaConfig
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers import LlamaTokenizerFast
from transformers import PreTrainedTokenizerFast
import logging

from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.models import Base_RelationExtraction
from medcat.utils.relation_extraction.models import BertModel_RelationExtraction
from medcat.utils.relation_extraction.models import ModernBertModel_RelationExtraction
from medcat.utils.relation_extraction.models import LlamaModel_RelationExtraction
from medcat.utils.relation_extraction.ml_utils import create_tokenizer_pretrain


logger = logging.getLogger(__name__)


class BaseTokenizerWrapper(PreTrainedTokenizerFast, ABC):

    def __init__(self, hf_tokenizers=None, max_seq_length: Optional[int] = None, add_special_tokens: Optional[bool] = False):
        self.hf_tokenizers = hf_tokenizers
        self.max_seq_length = max_seq_length
        self.add_special_tokens = add_special_tokens

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

    def get_size(self):
        return len(self.hf_tokenizers.vocab)

    def token_to_id(self, token):
        return self.hf_tokenizers.convert_tokens_to_ids(token)

    def get_pad_id(self):
        return self.hf_tokenizers.pad_token_id

    def __call__(self, text, truncation: Optional[bool] = True):
        if isinstance(text, str):
            result = self.hf_tokenizers.encode_plus(text, return_offsets_mapping=True, return_length=True, return_token_type_ids=True, return_attention_mask=True,
                                                    add_special_tokens=self.add_special_tokens, max_length=self.max_seq_length, padding="longest", truncation=truncation)

            return {'offset_mapping': result['offset_mapping'],
                    'input_ids': result['input_ids'],
                    'tokens':  self.hf_tokenizers.convert_ids_to_tokens(result['input_ids']),
                    'token_type_ids': result['token_type_ids'],
                    'attention_mask': result['attention_mask'],
                    'length': result['length']
                    }
        elif isinstance(text, list):
            results = self.hf_tokenizers._batch_encode_plus(text, return_offsets_mapping=True, return_length=True, return_token_type_ids=True,
                                                            add_special_tokens=self.add_special_tokens, max_length=self.max_seq_length,truncation=truncation)
            output = []
            for ind in range(len(results['input_ids'])):
                output.append({
                    'offset_mapping': results['offset_mapping'][ind],
                    'input_ids': results['input_ids'][ind],
                    'tokens':  self.hf_tokenizers.convert_ids_to_tokens(results['input_ids'][ind]),
                    'token_type_ids': results['token_type_ids'][ind],
                    'attention_mask': results['attention_mask'][ind],
                    'length': result['length']
                })
            return output
        else:
            raise Exception(
                "Unsupported input type, supported: text/list, but got: {}".format(type(text)))

    def save(self, dir_path):
        path = os.path.join(dir_path, self.name)
        self.hf_tokenizers.save_pretrained(path)


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


class TokenizerWrapperModernBERT(BaseTokenizerWrapper):
    ''' Wrapper around a huggingface ModernBERT tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.PreTrainedTokenizerFast`):
            A huggingface Fast tokenizer.
    '''
    name = 'modern-bert-tokenizer'
    pretrained_model_name_or_path = "answerdotai/ModernBERT-base"

    @abstractmethod
    def config_from_pretrained(self) -> PretrainedConfig:
        return BertConfig.from_pretrained(self.pretrained_model_name_or_path)

    @abstractmethod
    def config_from_json_file(self, file_path: str) -> PretrainedConfig:
        return ModernBertConfig.from_json_file(file_path)

    @abstractmethod
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


class TokenizerWrapperLlama(BaseTokenizerWrapper):
    ''' Wrapper around a huggingface Llama tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.LlamaTokenizerFast`):
            A huggingface Fast Llama.
    '''
    name = 'llama-tokenizer'
    pretrained_model_name_or_path = "meta-llama/Llama-3.1-8B"

    @abstractmethod
    def config_from_pretrained(self) -> PretrainedConfig:
        pass # perhaps some doc string

    @abstractmethod
    def config_from_json_file(self, file_path: str) -> PretrainedConfig:
        return LlamaConfig.from_json_file(file_path)

    @abstractmethod
    def model_from_pretrained(self, relcat_config: ConfigRelCAT, model_config: PretrainedConfig,
            pretrained_model_name_or_path: str = 'default') -> Base_RelationExtraction:
        if pretrained_model_name_or_path == 'default':
            pretrained_model_name_or_path = self.pretrained_model_name_or_path
        return LlamaModel_RelationExtraction(pretrained_model_name_or_path, relcat_config, model_config)

    def __init__(self, hf_tokenizers=None, max_seq_length: Optional[int] = None, add_special_tokens: Optional[bool] = False):
        self.hf_tokenizers = hf_tokenizers
        self.max_seq_length = max_seq_length
        self.add_special_tokens = add_special_tokens

    def __call__(self, text, truncation: Optional[bool] = True):
        if isinstance(text, str):
            result = self.hf_tokenizers.encode_plus(text, return_offsets_mapping=True, return_length=True, return_token_type_ids=True, return_attention_mask=True,
                                                    add_special_tokens=self.add_special_tokens, max_length=self.max_seq_length, padding="longest", truncation=truncation)

            return {'offset_mapping': result['offset_mapping'],
                    'input_ids': result['input_ids'],
                    'tokens':  self.hf_tokenizers.convert_ids_to_tokens(result['input_ids']),
                    'token_type_ids': result['token_type_ids'],
                    'attention_mask': result['attention_mask'],
                    'length': result['length']
                    }
        elif isinstance(text, list):
            results = self.hf_tokenizers._batch_encode_plus(text, return_offsets_mapping=True, return_length=True, return_token_type_ids=True,
                                                            add_special_tokens=self.add_special_tokens, max_length=self.max_seq_length,truncation=truncation)
            output = []
            for ind in range(len(results['input_ids'])):
                output.append({
                    'offset_mapping': results['offset_mapping'][ind],
                    'input_ids': results['input_ids'][ind],
                    'tokens':  self.hf_tokenizers.convert_ids_to_tokens(results['input_ids'][ind]),
                    'token_type_ids': results['token_type_ids'][ind],
                    'attention_mask': results['attention_mask'][ind],
                    'length': result['length']
                })
            return output
        else:
            raise Exception(
                "Unsuported input type, supported: text/list, but got: {}".format(type(text)))

    def save(self, dir_path):
        path = os.path.join(dir_path, self.name)
        self.hf_tokenizers.save_pretrained(path)

    @classmethod
    def load(cls, dir_path, **kwargs):
        tokenizer = cls()
        path = os.path.join(dir_path, cls.name)
        tokenizer.hf_tokenizers = LlamaTokenizerFast.from_pretrained(
            path, **kwargs)

        return tokenizer

    def get_size(self):
        return len(self.hf_tokenizers.vocab)

    def token_to_id(self, token):
        return self.hf_tokenizers.convert_tokens_to_ids(token)

    def get_pad_id(self):
        return self.hf_tokenizers.pad_token_id


def load_tokenizer(tokenizer_path: str,
                   config: ConfigRelCAT) -> BaseTokenizerWrapper:
    if os.path.exists(tokenizer_path):
        if "modern-bert-tokenizer" in config.general.tokenizer_name:
            tokenizer = TokenizerWrapperModernBERT.load(tokenizer_path)
        elif "bert" in config.general.tokenizer_name:
            tokenizer = TokenizerWrapperBERT.load(tokenizer_path)
        elif "llama" in config.general.tokenizer_name:
            tokenizer = TokenizerWrapperLlama.load(tokenizer_path)
        logger.info("Tokenizer loaded " + str(tokenizer.__class__.__name__) + " from:" + tokenizer_path)
        return tokenizer
    elif config.general.model_name:
        logger.info("Attempted to load Tokenizer from path:" + tokenizer_path +
                ", but it doesn't exist, loading default toknizer from model_name config.general.model_name:" + config.general.model_name)
        tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config.general.model_name),
                                            max_seq_length=config.general.max_seq_length,
                                            add_special_tokens=config.general.tokenizer_special_tokens
                                            )
        return create_tokenizer_pretrain(tokenizer, tokenizer_path)
    else:
        logger.info("Attempted to load Tokenizer from path:" + tokenizer_path +
                ", but it doesn't exist, loading default toknizer from model_name config.general.model_name:bert-base-uncased")
        return TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased"),
                                            max_seq_length=config.general.max_seq_length,
                                            add_special_tokens=config.general.tokenizer_special_tokens
                                            )
