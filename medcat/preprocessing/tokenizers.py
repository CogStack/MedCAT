import re
import os
import spacy
from typing import Any, List, Dict, cast, Iterable, Union, Pattern
from spacy.tokenizer import Tokenizer
from spacy.language import Language
from spacy.tokens import Doc
from tokenizers import ByteLevelBPETokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from medcat.config import Config


def spacy_extended(nlp: Language) -> Tokenizer:
    infix_re_list = ('\\.\\.+',
    '''(?<=[A-Za-z]{1})[\-_;\,\/~]+(?=[A-Za-z]{1})|(?<=[0-9]{1})[\-_;\,\/]+(?=[A-Za-z]{1})|(?<=[A-Za-z]{1})[\-_;\,\/]+(?=[0-9]{1})|\d{2,4}[\-\s_\*]\d{1,2}[\-\s_\*]\d{1,2}|\d{1,2}:\d{1,2}:\d{1,2}|\d{1,2}:\d{2}'''
     '…',
     '[\\p{So}]',
     '(?<=[[[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])\\.(?=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])',
     '(?<=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]]),(?=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])',
     '(?<=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])[?";:=,.]*(?:-|–|—|--|---|——|~)(?=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])',
     '(?<=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]"])[:<>=/](?=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])')

    prefix_iter = cast(Iterable[Union[str, Pattern[Any]]], Language.Defaults.prefixes)
    suffix_iter = cast(Iterable[Union[str, Pattern[Any]]], Language.Defaults.suffixes)
    prefix_re = spacy.util.compile_prefix_regex(prefix_iter)
    suffix_re = spacy.util.compile_suffix_regex(suffix_iter)
    infix_re = spacy.util.compile_infix_regex(infix_re_list)


    return Tokenizer(nlp.vocab,
            rules={},
            prefix_search=prefix_re.search,
            suffix_search=suffix_re.search,
            infix_finditer=infix_re.finditer
            )


def spacy_split_all(nlp: Language, config: Config) -> Tokenizer:

    token_characters = r'[^A-Za-z0-9\@]'

    if config.general.diacritics:
        token_characters = r'[^A-Za-zÀ-ÖØ-öø-ÿ0-9\@]'

    infix_re = re.compile(token_characters)
    suffix_re = re.compile(token_characters + r'$')
    prefix_re = re.compile(r'^' + token_characters)
    return Tokenizer(nlp.vocab,
            rules={},
            token_match=None,
            prefix_search=prefix_re.search,
            suffix_search=suffix_re.search,
            infix_finditer=infix_re.finditer
            )


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab: Any, unk_token: str = "[UNK]", max_input_chars_per_word: int = 200) -> None:
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> List:
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]
        Args:
          text: A single token or whitespace separated tokens. This should have.
            already been passed through `BasicTokenizer`.
        Returns:
          List: A list of wordpiece tokens.
        """

        # Why is convert_to_unicode undefined?
        text = convert_to_unicode(text) # type: ignore # noqa

        output_tokens = []

        # Why is whitespace_tokenize undefined?
        for token in whitespace_tokenize(text): # type: ignore # noqa
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class SpacyHFTok(object):

    def __init__(self, w2v: Any) -> None:
        self.nlp = spacy.load('en_core_sci_md', disable=['ner', 'parser'])
        self.emb_map = {}
        self.embs: List = []
        for key in w2v.wv.key_to_index.keys():
            self.emb_map[key] = len(self.embs)
            self.embs.append(w2v[key])

        # Add pad
        self.embs.append([0.0] * 300)

    def encode(self, text: str) -> 'SpacyHFDoc':
        doc = self.nlp(text)
        return SpacyHFDoc(doc)

    def token_to_id(self, tok: Any) -> Any:
        return self.emb_map.get(tok, len(self.emb_map) - 1)


class SpacyHFDoc(object):
    def __init__(self, doc: Doc) -> None:
        self.doc = doc
        self.tokens = [x.text for x in self.doc]
        self.offsets = [(x.idx, x.idx+len(x.text)) for x in self.doc]


class TokenizerWrapperBPE(object):

    def __init__(self, hf_tokenizers: Any) -> None:
        self.hf_tokenizers = hf_tokenizers

    def __call__(self, text: str) -> Dict:
        res = self.hf_tokenizers.encode(text)

        return {'offset_mapping': res.offsets,
                'input_ids': res.ids,
                'tokens': res.tokens,
                }

    def save(self, dir_path, name='bbpe'):
        self.hf_tokenizers.save_model(dir_path, prefix=name)

    @classmethod
    def load(cls, dir_path, name='bbpe', **kwargs):
        tokenizer = cls()
        vocab_file = os.path.join(dir_path, f'{name}-vocab.json')
        merges_file = os.path.join(dir_path, f'{name}-merges.txt')
        tokenizer.hf_tokenizers = ByteLevelBPETokenizer.from_file(vocab_filename=vocab_file,
                                                                  merges_filename=merges_file,
                                                                  **kwargs)

        return tokenizer


class TokenizerWrapperBERT(object):

    def __init__(self, hf_tokenizers=None):
        self.hf_tokenizers = hf_tokenizers

    def __call__(self, text: str) -> Dict:
        res = self.hf_tokenizers.encode_plus(text,
                return_offsets_mapping=True, add_special_tokens=False)

        return {'offset_mapping': res['offset_mapping'],
                'input_ids': res['input_ids'],
                'tokens':  self.hf_tokenizers.convert_ids_to_tokens(res['input_ids']),
                }

    def save(self, dir_path: str, name: str='bert') -> None:
        path = os.path.join(dir_path, name)
        self.hf_tokenizers.save_pretrained(path)

    @classmethod
    def load(cls, dir_path: str, name: str = 'bert', **kwargs) -> Any:
        tokenizer = cls()
        path = os.path.join(dir_path, name)
        tokenizer.hf_tokenizers = BertTokenizerFast.from_pretrained(path, **kwargs)

        return tokenizer
