import spacy
from spacy.tokenizer import Tokenizer
from spacy.language import Language
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
