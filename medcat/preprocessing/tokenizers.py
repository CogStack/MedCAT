import spacy
from spacy.tokenizer import Tokenizer
from spacy.language import Language
from tokenizers import ByteLevelBPETokenizer
import re

def spacy_extended(nlp):
    infix_re_list = ('\\.\\.+',
    '''(?<=[A-Za-z]{1})[\-_;\,\/~]+(?=[A-Za-z]{1})|(?<=[0-9]{1})[\-_;\,\/]+(?=[A-Za-z]{1})|(?<=[A-Za-z]{1})[\-_;\,\/]+(?=[0-9]{1})|\d{2,4}[\-\s_\*]\d{1,2}[\-\s_\*]\d{1,2}|\d{1,2}:\d{1,2}:\d{1,2}|\d{1,2}:\d{2}'''
     '…',
     '[\\p{So}]',
     '(?<=[[[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])\\.(?=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])',
     '(?<=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]]),(?=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])',
     '(?<=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])[?";:=,.]*(?:-|–|—|--|---|——|~)(?=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])',
     '(?<=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]"])[:<>=/](?=[[[\\p{Lu}&&\\p{Latin}]||[ЁА-Я]||[ӘӨҮҖҢҺ]||[Α-ΩΆΈΊΌΏΉΎ]||[\\p{Ll}&&\\p{Latin}]||[ёа-я]||[әөүҗңһ]||[α-ωάέίόώήύ]||[\\p{L}&&\\p{Bengali}]||[\\p{L}&&\\p{Hebrew}]||[\\p{L}&&\\p{Arabic}]||[\\p{L}&&\\p{Sinhala}]]])')

    prefix_re = spacy.util.compile_prefix_regex(Language.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(Language.Defaults.suffixes)
    infix_re = spacy.util.compile_infix_regex(infix_re_list)


    return Tokenizer(nlp.vocab,
            rules={},
            prefix_search=prefix_re.search,
            suffix_search=suffix_re.search,
            infix_finditer=infix_re.finditer
            )


def spacy_split_all(nlp):
    infix_re = re.compile(r'''[^A-Za-z0-9\@]''')
    suffix_re = re.compile(r'''[^A-Za-z0-9\@]$''')
    prefix_re = re.compile(r'''^[^A-Za-z0-9\@]''')
    return Tokenizer(nlp.vocab,
            rules={},
            token_match=None,
            prefix_search=prefix_re.search,
            suffix_search=suffix_re.search,
            infix_finditer=infix_re.finditer
            )


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.
    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.
    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]
    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.
    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
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
    import spacy
    import numpy as np

    def __init__(self, w2v):
        self.nlp = spacy.load('en_core_sci_md', disable=['ner', 'parser'])
        self.emb_map = {}
        self.embs = []
        for key in w2v.wv.vocab.keys():
            self.emb_map[key] = len(self.embs)
            self.embs.append(w2v[key])

        # Add pad
        self.embs.append([0.0] * 300)

    def encode(self, text):
        doc = self.nlp(text)
        return SpacyHFDoc(doc)

    def token_to_id(self, tok):
        return self.emb_map.get(tok, len(self.emb_map) - 1)


class SpacyHFDoc(object):
    def __init__(self, doc):
        self.doc = doc
        self.tokens = [x.text for x in self.doc]
        self.offsets = [(x.idx, x.idx+len(x.text)) for x in self.doc]


class TokenizerWrapperBPE(object):
    '''
    '''

    def __init__(self, hf_tokenizers=None):
        self.hf_tokenizers = hf_tokenizers


    def __call__(self, text):
        res = self.hf_tokenizers.encode(text)

        return {'offset_mapping': res.offsets,
                'input_ids': res.ids,
                'tokens': res.tokens,
                }


    def save(self, dir_path, name='bbpe'):
        self.hf_tokenizers.save_model(dir_path, name=name)

    @classmethod
    def load(cls, dir_path, name='bbpe', lowercase=True):
        tokenizer = cls()
        vocab_file = dir_path + "{}-vocab.json".format(name)
        merges_file = dir_path + "{}-merges.txt".format(name)
        tokenizer.hf_tokenizers = ByteLevelBPETokenizer.from_file(vocab_filename=vocab_file, merges_filename=merges_file, lowercase=lowercase)

        return tokenizer
