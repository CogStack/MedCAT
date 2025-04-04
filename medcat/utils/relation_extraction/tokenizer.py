import os
from typing import Optional
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
import logging

from medcat.config_rel_cat import ConfigRelCAT


logger = logging.getLogger(__name__)


class BaseTokenizerWrapper(PreTrainedTokenizerFast):

    def __init__(self, hf_tokenizers=None, max_seq_length: Optional[int] = None, add_special_tokens: Optional[bool] = False):
        self.hf_tokenizers = hf_tokenizers
        self.max_seq_length = max_seq_length
        self.add_special_tokens = add_special_tokens

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


def load_default_tokenizer(tokenizer_path: str,
                           config: ConfigRelCAT) -> BaseTokenizerWrapper:
    if config.general.model_name:
        logger.info("Attempted to load Tokenizer from path:" + tokenizer_path +
                ", but it doesn't exist, loading default toknizer from model_name config.general.model_name:" + config.general.model_name)
        from medcat.utils.relation_extraction.bert.tokenizer import TokenizerWrapperBERT
        tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config.general.model_name),
                                            max_seq_length=config.general.max_seq_length,
                                            add_special_tokens=config.general.tokenizer_special_tokens
                                            )
        # import dynamically, only if needed
        from medcat.utils.relation_extraction.ml_utils import create_tokenizer_pretrain
        return create_tokenizer_pretrain(tokenizer, tokenizer_path)
    else:
        logger.info("Attempted to load Tokenizer from path:" + tokenizer_path +
                ", but it doesn't exist, loading default toknizer from model_name config.general.model_name:bert-base-uncased")
        from medcat.utils.relation_extraction.bert.tokenizer import TokenizerWrapperBERT
        return TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased"),
                                            max_seq_length=config.general.max_seq_length,
                                            add_special_tokens=config.general.tokenizer_special_tokens
                                            )
