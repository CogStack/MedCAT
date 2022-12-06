import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, overload
from tokenizers import Tokenizer, ByteLevelBPETokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast


class TokenizerWrapperBase(ABC):

    name: str

    def __init__(self, hf_tokenizer: Optional[Tokenizer] = None) -> None:
        self.hf_tokenizers = hf_tokenizer

    @overload
    def __call__(self, text: str) -> Dict: ...

    @overload
    def __call__(self, text: List[str]) -> List[Dict]: ...

    @abstractmethod
    def __call__(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]: ...

    @abstractmethod
    def save(self, dir_path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, dir_path: str, **kwargs) -> Tokenizer: ...

    @abstractmethod
    def get_size(self) -> int: ...

    @abstractmethod
    def token_to_id(self, token: str) -> Union[int, List[int]]: ...

    @abstractmethod
    def get_pad_id(self) -> Union[Optional[int], List[int]]: ...

    def ensure_tokenizer(self) -> Tokenizer:
        if self.hf_tokenizers is None:
            raise ValueError("The tokenizer is not loaded yet")
        return self.hf_tokenizers


class TokenizerWrapperBPE(TokenizerWrapperBase):
    """Wrapper around a huggingface tokenizer so that it works with the
    MetaCAT models.

    Args:
        tokenizers.ByteLevelBPETokenizer:
            A huggingface BBPE tokenizer.
    """
    name = 'bbpe'

    def __init__(self, hf_tokenizers: Optional[ByteLevelBPETokenizer] = None) -> None:
        super().__init__(hf_tokenizers)

        if self.hf_tokenizers is not None:
            # For whatever reason added tokens do not persist with this tokenizer, what to do
            self.hf_tokenizers.add_tokens(['<PAD>'])

    @overload
    def __call__(self, text: str) -> Dict: ...

    @overload
    def __call__(self, text: List[str]) -> List[Dict]: ...

    def __call__(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """Tokenize some text

        Args:
            text (Union(str, List[str])):
                Text/texts to be tokenized.

        Returns:
            Union(dict, List[dict]):
                Dictionary/ies containing `offset_mapping`, `input_ids` and `tokens` corresponding to the
                input text/s.
        """
        self.hf_tokenizers = self.ensure_tokenizer()

        if isinstance(text, str):
            result = self.hf_tokenizers.encode(text)

            return {'offset_mapping': result.offsets,
                    'input_ids': result.ids,
                    'tokens': result.tokens,
                    }
        elif isinstance(text, list):
            results = self.hf_tokenizers.encode_batch(text)
            output = []
            for result in results:
                output.append({'offset_mapping': result.offsets,
                    'input_ids': result.ids,
                    'tokens': result.tokens,
                })

            return output
        else:
            raise Exception("Unsuported input type, supported: text/list, but got: {}".format(type(text)))

    def save(self, dir_path: str) -> None:
        self.hf_tokenizers = self.ensure_tokenizer()

        if self.hf_tokenizers is None:
            raise ValueError("The tokenizer is not loaded yet")

        self.hf_tokenizers.save_model(dir_path, prefix=self.name)

    @classmethod
    def load(cls, dir_path: str, **kwargs) -> "TokenizerWrapperBPE":
        tokenizer = cls()
        vocab_file = os.path.join(dir_path, f'{tokenizer.name}-vocab.json')
        merges_file = os.path.join(dir_path, f'{tokenizer.name}-merges.txt')
        tokenizer.hf_tokenizers = ByteLevelBPETokenizer.from_file(vocab_filename=vocab_file,
                                                                  merges_filename=merges_file,
                                                                  **kwargs)
        # For whatever reason added tokens do not persist with this tokenizer, so we added it at each load
        tokenizer.hf_tokenizers.add_tokens(['<PAD>'])
        return tokenizer

    def get_size(self) -> int:
        self.hf_tokenizers = self.ensure_tokenizer()
        return self.hf_tokenizers.get_vocab_size()

    def token_to_id(self, token: str) -> Union[int, List[int]]:
        self.hf_tokenizers = self.ensure_tokenizer()
        return self.hf_tokenizers.token_to_id(token)

    def get_pad_id(self) -> Union[int, List[int]]:
        pad = self.token_to_id('<PAD>')
        if pad is None:
            raise Exception("No <PAD> token in the vocabulary of the tokenizer, please add it")
        return pad


class TokenizerWrapperBERT(TokenizerWrapperBase):
    """Wrapper around a huggingface BERT tokenizer so that it works with the
    MetaCAT models.

    Args:
        transformers.models.bert.tokenization_bert_fast.BertTokenizerFast:
            A huggingface Fast BERT.
    """
    name = 'bert-tokenizer'

    def __init__(self, hf_tokenizers: Optional[BertTokenizerFast] = None) -> None:
        super().__init__(hf_tokenizers)

    @overload
    def __call__(self, text: str) -> Dict: ...

    @overload
    def __call__(self, text: List[str]) -> List[Dict]: ...

    def __call__(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        self.hf_tokenizers = self.ensure_tokenizer()
        if isinstance(text, str):
            result = self.hf_tokenizers.encode_plus(text, return_offsets_mapping=True,
                    add_special_tokens=False)

            return {'offset_mapping': result['offset_mapping'],
                    'input_ids': result['input_ids'],
                    'tokens':  self.hf_tokenizers.convert_ids_to_tokens(result['input_ids']),
                    }
        elif isinstance(text, list):
            results = self.hf_tokenizers._batch_encode_plus(text, return_offsets_mapping=True,
                    add_special_tokens=False)
            output = []
            for ind in range(len(results['input_ids'])):
                output.append({'offset_mapping': results['offset_mapping'][ind],
                    'input_ids': results['input_ids'][ind],
                    'tokens':  self.hf_tokenizers.convert_ids_to_tokens(results['input_ids'][ind]),
                    })
            return output
        else:
            raise Exception("Unsuported input type, supported: text/list, but got: {}".format(type(text)))

    def save(self, dir_path: str) -> None:
        self.hf_tokenizers = self.ensure_tokenizer()
        path = os.path.join(dir_path, self.name)
        self.hf_tokenizers.save_pretrained(path)

    @classmethod
    def load(cls, dir_path: str, **kwargs) -> "TokenizerWrapperBERT":
        tokenizer = cls()
        path = os.path.join(dir_path, cls.name)
        tokenizer.hf_tokenizers = BertTokenizerFast.from_pretrained(path, **kwargs)

        return tokenizer

    def get_size(self) -> int:
        self.hf_tokenizers = self.ensure_tokenizer()
        return len(self.hf_tokenizers.vocab)

    def token_to_id(self, token: str) -> Union[int, List[int]]:
        self.hf_tokenizers = self.ensure_tokenizer()
        return self.hf_tokenizers.convert_tokens_to_ids(token)

    def get_pad_id(self) -> Optional[int]:
        self.hf_tokenizers = self.ensure_tokenizer()
        return self.hf_tokenizers.pad_token_id
