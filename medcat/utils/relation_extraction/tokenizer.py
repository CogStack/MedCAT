import os
from typing import Optional
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class TokenizerWrapperBERT(BertTokenizerFast):
    ''' Wrapper around a huggingface BERT tokenizer so that it works with the
    RelCAT models.

    Args:
        hf_tokenizers (`transformers.models.bert.tokenization_bert_fast.BertTokenizerFast`):
            A huggingface Fast BERT.
    '''
    name = 'bert-tokenizer'

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
                "Unsupported input type, supported: text/list, but got: {}".format(type(text)))

    def save(self, dir_path):
        path = os.path.join(dir_path, self.name)
        self.hf_tokenizers.save_pretrained(path)

    @classmethod
    def load(cls, dir_path, **kwargs):
        tokenizer = cls()
        path = os.path.join(dir_path, cls.name)
        tokenizer.hf_tokenizers = BertTokenizerFast.from_pretrained(
            path, **kwargs)

        return tokenizer

    def get_size(self):
        return len(self.hf_tokenizers.vocab)

    def token_to_id(self, token):
        return self.hf_tokenizers.convert_tokens_to_ids(token)

    def get_pad_id(self):
        return self.hf_tokenizers.pad_token_id
