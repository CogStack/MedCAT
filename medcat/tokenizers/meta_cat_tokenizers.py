from tokenizers import ByteLevelBPETokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
import os


class TokenizerWrapperBPE(object):
    ''' Wrapper around a huggingface tokenizer so that it works with the
    MetaCAT models.

    Args:
        hf_tokenizers (`tokenizers.ByteLevelBPETokenizer`):
            A huggingface BBPE tokenizer.
    '''
    name = 'bbpe'

    def __init__(self, hf_tokenizers=None):
        self.hf_tokenizers = hf_tokenizers


    def __call__(self, text):
        r''' Tokenize some text

        Args:
            text (`str`):
                Text to be tokenized.

        Returns:
            res (`dict`):
                Dictionary containing `offset_mapping`, `input_ids` and `tokens` corresponding to the
                input text.

        '''
        res = self.hf_tokenizers.encode(text)

        return {'offset_mapping': res.offsets,
                'input_ids': res.ids,
                'tokens': res.tokens,
                }


    def save(self, dir_path):
        self.hf_tokenizers.save_model(dir_path, prefix=self.name)


    @classmethod
    def load(cls, dir_path, **kwargs):
        tokenizer = cls()
        vocab_file = os.path.join(dir_path, f'{tokenizer.name}-vocab.json')
        merges_file = os.path.join(dir_path, f'{tokenizer.name}-merges.txt')
        tokenizer.hf_tokenizers = ByteLevelBPETokenizer.from_file(vocab_filename=vocab_file,
                                                                  merges_filename=merges_file,
                                                                  **kwargs)
        return tokenizer


class TokenizerWrapperBERT(object):
    ''' Wrapper around a huggingface BERT tokenizer so that it works with the
    MetaCAT models.

    Args:
        hf_tokenizers (`transformers.models.bert.tokenization_bert_fast.BertTokenizerFast`):
            A huggingface Fast BERT.
    '''
    name = 'bert'
    def __init__(self, hf_tokenizers=None):
        self.hf_tokenizers = hf_tokenizers


    def __call__(self, text):
        res = self.hf_tokenizers.encode_plus(text,
                return_offsets_mapping=True, add_special_tokens=False)

        return {'offset_mapping': res['offset_mapping'],
                'input_ids': res['input_ids'],
                'tokens':  self.hf_tokenizers.convert_ids_to_tokens(res['input_ids']),
                }


    def save(self, dir_path):
        path = os.path.join(dir_path, self.name)
        self.hf_tokenizers.save_pretrained(path)


    @classmethod
    def load(cls, dir_path, **kwargs):
        tokenizer = cls()
        path = os.path.join(dir_path, self.name)
        tokenizer.hf_tokenizers = BertTokenizerFast.from_pretrained(path, **kwargs)

        return tokenizer
