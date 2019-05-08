from medcat.preprocessing.cleaners import spacy_tag_punct
from spacy.tokens import Token, Doc, Span
from medcat.utils.spelling import SpacySpellChecker
import spacy
import os

class SpacyPipe(object):
    SPACY_MODEL = os.getenv("SPACY_MODEL", 'en_core_sci_md')
    def __init__(self, tokenizer, disable=['ner', 'parser', 'vectors', 'textcat']):
        self.nlp = spacy.load(self.SPACY_MODEL, disable=disable)
        self.nlp.tokenizer = tokenizer(self.nlp)


    def add_punct_tagger(self, tagger):
        """ Tagging for punct
        """
        self.nlp.add_pipe(tagger, name='tag_punct', first=True)
        # Add custom fields needed for this usecase
        Token.set_extension('is_punct', default=False, force=True)
        Token.set_extension('to_skip', default=False, force=True)


    def add_spell_checker(self, spell_checker):
        spacy_spell_checker = SpacySpellChecker(spell_checker=spell_checker)
        self.nlp.add_pipe(spacy_spell_checker, name='spell_checker', last=True)

        # Add custom fields needed for this usecase
        Token.set_extension('verified', default=False, force=True)
        Token.set_extension('norm', default=None, force=True)
        Token.set_extension('lower', default=None, force=True)


    def add_cat(self, spacy_cat):
        self.nlp.add_pipe(spacy_cat, name='cat', last=True)

        # Add custom fields needed for this usecase
        Doc.set_extension('ents', default=None, force=True)
        Span.set_extension('acc', default=-1, force=True)
        Span.set_extension('cui', default=-1, force=True)
        Span.set_extension('tui', default=-1, force=True)


    def __call__(self, text):
        return self.nlp(text)
