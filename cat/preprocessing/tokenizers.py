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
    infix_re = re.compile(r'''[^A-Za-z0-9]''')
    suffix_re = re.compile(r'''[^A-Za-z0-9]$''')
    prefix_re = re.compile(r'''^[^A-Za-z0-9]''')
    return Tokenizer(nlp.vocab,
            rules={},
            token_match=None,
            prefix_search=prefix_re.search,
            suffix_search=suffix_re.search,
            infix_finditer=infix_re.finditer
            )
