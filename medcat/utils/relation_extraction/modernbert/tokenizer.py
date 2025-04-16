import os
from transformers import PreTrainedTokenizerFast
import logging

from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper_RelationExtraction

logger = logging.getLogger(__name__)


class TokenizerWrapperModernBERT_RelationExtraction(BaseTokenizerWrapper_RelationExtraction):
    ''' Wrapper around a huggingface ModernBERT tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.PreTrainedTokenizerFast`):
            A huggingface Fast tokenizer.
    '''
    name = "tokenizer_wrapper_modern_bert_rel"
    pretrained_model_name_or_path = "answerdotai/ModernBERT-base"

    @classmethod
    def load(cls, tokenizer_path: str, relcat_config: ConfigRelCAT, **kwargs) -> "TokenizerWrapperModernBERT_RelationExtraction":
        tokenizer = cls()
        path = os.path.join(tokenizer_path, cls.name)

        if tokenizer_path:
            tokenizer.hf_tokenizers = PreTrainedTokenizerFast.from_pretrained(
                path, **kwargs)
        else:
            relcat_config.general.model_name = cls.pretrained_model_name_or_path
            tokenizer.hf_tokenizers = PreTrainedTokenizerFast.from_pretrained(
                path=relcat_config.general.model_name)
        return tokenizer
