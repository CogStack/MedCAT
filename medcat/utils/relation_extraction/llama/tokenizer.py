import os
from typing import Optional
from transformers import LlamaTokenizerFast
import logging

from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper_RelationExtraction

logger = logging.getLogger(__name__)


class TokenizerWrapperLlama_RelationExtraction(BaseTokenizerWrapper_RelationExtraction):
    ''' Wrapper around a huggingface Llama tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.LlamaTokenizerFast`):
            A huggingface Fast Llama.
    '''
    name = 'llama-tokenizer'
    pretrained_model_name_or_path = "meta-llama/Llama-3.1-8B"

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

    @classmethod
    def load(cls, tokenizer_path: str, relcat_config: ConfigRelCAT, **kwargs) -> "TokenizerWrapperLlama_RelationExtraction":
        tokenizer = cls()
        path = os.path.join(tokenizer_path, cls.name)

        if tokenizer_path:
            tokenizer.hf_tokenizers = LlamaTokenizerFast.from_pretrained(
                path, **kwargs)
        else:
            relcat_config.general.model_name = cls.pretrained_model_name_or_path
            tokenizer.hf_tokenizers = LlamaTokenizerFast.from_pretrained(
                path=relcat_config.general.model_name)
        return tokenizer
