import os
from transformers import PreTrainedTokenizerFast
import logging

from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper


logger = logging.getLogger(__name__)


class TokenizerWrapperModernBERT(BaseTokenizerWrapper):
    ''' Wrapper around a huggingface ModernBERT tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.PreTrainedTokenizerFast`):
            A huggingface Fast tokenizer.
    '''
    name = 'modern-bert-tokenizer'

    @classmethod
    def load(cls, dir_path, **kwargs):
        tokenizer = cls()
        path = os.path.join(dir_path, cls.name)
        tokenizer.hf_tokenizers = PreTrainedTokenizerFast.from_pretrained(
            path, **kwargs)

        return tokenizer
