import re

def tag_skip_and_punct(doc, config):
    r''' Detects and tags spacy tokens that are punctuation and that should be skipped.

    Args:
        doc (`spacy.tokens.Doc`):
            Spacy document that will be tagged.
        config (`medcat.config.Config`):
            Global config for medcat.

    Return:
        (`spacy.tokens.Doc):
            Tagged spacy document
    '''
    # Make life easier
    cnf_p = config.preprocessing

    for token in doc:
        if config.punct_checker.match(token.lower_) and token.text not in cnf_p['keep_punct']:
            # There can't be punct in a token if it also has text
            token._.is_punct = True
            token._.to_skip = True
        elif config.word_skipper.match(token.lower_):
            # Skip if specific strings
            token._.to_skip = True
        elif cnf_p['skip_stopwords'] and token.is_stop:
            token._.to_skip = True

    return doc
