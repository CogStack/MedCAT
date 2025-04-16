import os
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import logging

from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper_RelationExtraction

logger = logging.getLogger(__name__)


class TokenizerWrapperBERT_RelationExtraction(BaseTokenizerWrapper_RelationExtraction):
    ''' Wrapper around a huggingface BERT tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.models.bert.tokenization_bert_fast.PreTrainedTokenizerFast`):
            A huggingface Fast BERT.
    '''
    name = 'bert-tokenizer'
    pretrained_model_name_or_path = "bert-base-uncased"

    @classmethod
    def load(cls, dir_path: str, **kwargs):
        tokenizer = cls()
        path = os.path.join(dir_path, cls.name)

        if dir_path:
            tokenizer.hf_tokenizers = BertTokenizerFast.from_pretrained(
                path, **kwargs)
        else:
            tokenizer.hf_tokenizers = BertTokenizerFast.from_pretrained(
                path=cls.pretrained_model_name_or_path)

        return tokenizer
