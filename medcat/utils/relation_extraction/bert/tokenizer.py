import os
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import logging

from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper


logger = logging.getLogger(__name__)


class TokenizerWrapperBERT(BaseTokenizerWrapper):
    ''' Wrapper around a huggingface BERT tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.models.bert.tokenization_bert_fast.PreTrainedTokenizerFast`):
            A huggingface Fast BERT.
    '''
    name = 'bert-tokenizer'

    @classmethod
    def load(cls, dir_path, **kwargs):
        tokenizer = cls()
        path = os.path.join(dir_path, cls.name)
        tokenizer.hf_tokenizers = BertTokenizerFast.from_pretrained(
            path, **kwargs)

        return tokenizer
