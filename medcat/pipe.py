from spacy.tokens import Token, Doc, Span
from medcat.utils.normalizers import TokenNormalizer
import spacy
import os

class Pipe(object):
    r''' A wrapper around the standard spacy pipeline.

    Args:
        tokenizer (`spacy.tokenizer.Tokenizer`):
            What will be used to split text into tokens, can be anything built as a spacy tokenizer.
        config (`medcat.config.Config`):
            Global config for medcat.

    Properties:
        nlp (spacy.language.<lng>):
            The base spacy NLP pipeline.
    '''
    def __init__(self, tokenizer, config):
        self.nlp = spacy.load(config.general['spacy_model'], disable=config.general['spacy_disabled_components'])
        if config.preprocessing['stopwords'] is not None:
            self.nlp.Defaults.stop_words = set(config.preprocessing['stopwords'])
        self.nlp.tokenizer = tokenizer(self.nlp)


    def add_tagger(self, tagger, name, additional_fields=[]):
        r''' Add any kind of a tagger for tokens.

        Args:
            tagger (`object/function`):
                Any object/function that takes a spacy doc as an input, does something
                and returns the same doc.
            name (`str`):
                Name for this component in the pipeline.
            additional_fields (`List[str]`):
                Fields to be added to the `_` properties of a token.
        '''
        self.nlp.add_pipe(tagger, name='tag_' + name, first=True)
        # Add custom fields needed for this usecase
        Token.set_extension('to_skip', default=False, force=True)

        # Add any additional fields that are required
        for field in additional_fields:
            Token.set_extension(field, default=False, force=True)


    def add_token_normalizer(self, config, spell_checker=None):
        token_normalizer = TokenNormalizer(spell_checker=spell_checker, config=config)
        self.nlp.add_pipe(token_normalizer, name='token_normalizer', last=True)

        # Add custom fields needed for this usecase
        Token.set_extension('norm', default=None, force=True)


    def add_ner(self, ner):
        r''' Add NER from CAT to the pipeline, will also add the necessary fields
        to the document and Span objects.

        '''
        self.nlp.add_pipe(ner, name='cat_ner', last=True)

        Doc.set_extension('ents', default=[], force=True)
        Span.set_extension('confidence', default=-1, force=True)
        Span.set_extension('id', default=0, force=True)
        # Do not set this property if a vocabulary apporach is not used, this name must
        #refer to a name2cuis in the cdb.
        Span.set_extension('detected_name', default=None, force=True)
        Span.set_extension('link_candidates', default=None, force=True)


    def add_linker(self, linker):
        r''' Add entity linker to the pipeline, will also add the necessary fields
        to Span object.

        linker (object/function):
            Any object/function created based on the requirements for a spaCy pipeline components. Have
            a look at https://spacy.io/usage/processing-pipelines#custom-components
        '''
        self.nlp.add_pipe(linker, name='cat_linker', last=True)
        Span.set_extension('cui', default=-1, force=True)
        Span.set_extension('context_similarity', default=-1, force=True)


    def add_meta_cat(self, meta_cat, name):
        self.nlp.add_pipe(meta_cat, name=name, last=True)

        # Only the meta_anns field is needed, it will be a dictionary 
        #of {category_name: value, ...}
        Span.set_extension('meta_anns', default=None, force=True)


    def __call__(self, text):
        return self.nlp(text)
