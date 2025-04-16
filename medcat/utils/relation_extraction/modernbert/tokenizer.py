import os
from transformers import PreTrainedTokenizerFast
import logging

from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper_RelationExtraction

logger = logging.getLogger(__name__)


class TokenizerWrapperModernBERT_RelationExtraction(BaseTokenizerWrapper_RelationExtraction):
    ''' Wrapper around a huggingface ModernBERT tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.PreTrainedTokenizerFast`):
            A huggingface Fast tokenizer.
    '''
    name = 'modern-bert-tokenizer'
    pretrained_model_name_or_path = "answerdotai/ModernBERT-base"

    @classmethod
    def load(cls, dir_path, **kwargs):
        tokenizer = cls()
        path = os.path.join(dir_path, cls.name)

        if dir_path:
            tokenizer.hf_tokenizers = PreTrainedTokenizerFast.from_pretrained(
                path, **kwargs)
        else:
            tokenizer.hf_tokenizers = PreTrainedTokenizerFast.from_pretrained(
                path=cls.pretrained_model_name_or_path)
        return tokenizer
