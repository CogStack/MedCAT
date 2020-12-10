#import hunspell
import re
from collections import Counter
from spacy.tokens import Span
import spacy
import os

CONTAINS_NUMBER = re.compile('[0-9]+')


class BasicSpellChecker(object):
    r'''
    '''
    def __init__(self, cdb_vocab, config, data_vocab=None):
        self.vocab = cdb_vocab
        self.config = config
        self.data_vocab = data_vocab


    def P(self, word):
        "Probability of `word`."
	# use inverse of rank as proxy
	# returns 0 if the word isn't in the dictionary
        cnt = self.vocab.get(word, 0)
        if cnt != 0:
            return -1 / cnt
        else:
            return 0


    def __contains__(self, word):
        if word in self.vocab:
            return True
        elif self.data_vocab is not None and word in self.data_vocab:
            return False
        else:
            return False


    def fix(self, word):
        "Most probable spelling correction for word."
        fix = max(self.candidates(word), key=self.P)
        if fix != word:
            return fix
        else:
            return None

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        if self.config.general['spell_check_deep']:
            # This will check a two letter edit distance
            return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])
        else:
            # Will check only one letter edit distance
            return (self.known([word]) or self.known(self.edits1(word))  or [word])


    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.vocab)


    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)


    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))


    def edits3(self, word):
        "All edits that are two edits away from `word`."
        # Do d3 edits
        pass


class TokenNormalizer(object):
    r''' Will normalize all tokens in a spacy document.

    Args:
        config
        spell_checker
    '''

    def __init__(self, config, spell_checker=None):
        self.config = config
        self.spell_checker = spell_checker
        self.nlp = spacy.load(config.general['spacy_model'], disable=config.general['spacy_disabled_components'])

    def __call__(self, doc):
        for token in doc:
            if len(token.lower_) < self.config.preprocessing['min_len_normalize']:
                token._.norm = token.lower_
            elif token.lemma_ == '-PRON-':
                token._.norm = token.lemma_
                token._.to_skip = True
            else:
                token._.norm = token.lemma_.lower()

            if self.config.general['spell_check']:
                # Fix the token if necessary
                if len(token.text) >= self.config.general['spell_check_len_limit'] and not token._.is_punct \
                        and token.lower_ not in self.spell_checker and not CONTAINS_NUMBER.search(token.lower_):
                    fix = self.spell_checker.fix(token.lower_)
                    if fix is not None:
                        tmp = self.nlp(fix)[0]
                        if len(token.lower_) < self.config.preprocessing['min_len_normalize']:
                            token._.norm = tmp.lower_
                        else:
                            token._.norm = tmp.lemma_.lower()
        return doc
