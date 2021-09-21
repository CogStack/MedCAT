from medcat.pipeline.pipe_runner import PipeRunner


def tag_skip_and_punct(nlp, name, config):
    r''' Detects and tags spacy tokens that are punctuation and that should be skipped.

     Args:
         nlp (spacy.language.<lng>):
             The base spacy NLP pipeline.
         name (`str`):
             The component instance name.
         config (`medcat.config.Config`):
             Global config for medcat.
    '''
    return _Tagger(nlp, name, config)


# This is not elegant.
tag_skip_and_punct.name = "tag_skip_and_punct"


class _Tagger(PipeRunner):

    def __init__(self, nlp, name, config):
        self.name = name
        self.config = config
        super().__init__(self.config.general['workers'])

    def __call__(self, doc):
        # Make life easier
        cnf_p = self.config.preprocessing

        for token in doc:
            if self.config.punct_checker.match(token.lower_) and token.text not in cnf_p['keep_punct']:
                # There can't be punct in a token if it also has text
                token._.is_punct = True
                token._.to_skip = True
            elif self.config.word_skipper.match(token.lower_):
                # Skip if specific strings
                token._.to_skip = True
            elif cnf_p['skip_stopwords'] and token.is_stop:
                token._.to_skip = True

        return doc
