import pandas
import spacy
from spacy.tokenizer import Tokenizer
from cat.umls import UMLS
from cat.spacy_cat import SpacyCat
from preprocessing.tokenizers import spacy_split_all
from preprocessing.cleaners import spacy_tag_punct, clean_umls
from spacy.tokens import Token
from preprocessing.spelling import CustomSpellChecker, SpacySpellChecker
from preprocessing.spacy_pipe import SpacyPipe
from preprocessing.iterators import EmbMimicCSV
from gensim.models import FastText

class CAT(object):
    """ Annotate a dataset
    """
    def __init__(self, umls, vocab=None):
        self.umls = umls
        # Build the required spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all)
        self.nlp.add_punct_tagger(tagger=spacy_tag_punct)

        # Add spell checker pipe
        self.spell_checker = CustomSpellChecker(words=umls.vocab, big_vocab=vocab)
        self.nlp.add_spell_checker(spell_checker=self.spell_checker)

        # Add cat
        self.spacy_cat = SpacyCat(umls=umls, vocab=vocab)
        self.nlp.add_cat(spacy_cat=self.spacy_cat)


    def __call__(self, text):
        return self.nlp(text)
