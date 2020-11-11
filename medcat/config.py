import re
import logging

class Config(object):
    def __init__(self):
        # CDB Maker
        self.cdb_maker = {
                # Separator that will be used to merge tokens of a name.
                'separator': '',
                # If multiple names or type_ids for a concept present in one row of a CSV, they are separted
                #by the character below.
                'multi_separator': '|',
                # Name versions to be generated.
                'name_versions': ['CLEAN', 'LOWER'],

                }

        self.general = {
                # Log level for the whole medcat
                'log_level': logging.DEBUG,
                'log_format': '%(asctime)s: %(message)s',
                'spacy_disabled_components': ['tagger', 'ner', 'parser', 'vectors', 'textcat',
                                              'entity_linker', 'sentencizer', 'entity_ruler', 'merge_noun_chunks',
                                              'merge_entities', 'merge_subtokens'],
                # What model will be used for tokenization
                'spacy_model': 'en_core_sci_md',
                }

        self.preprocessing = {
                # Should stopwords be skipped/ingored when processing input
                'skip_stopwords': False,
                # This words will be completly ignored from concepts and from the text
                'words_to_skip': ['and', 'or', 'nos'],
                # All punct will be skipped by default, here you can set what will be kept
                'keep_punct': {'.', ':'},
                # Nothing below this length will ever be normalized (input tokens or concept names)
                'min_len_normalize': 4,
                }

        self.ner = {
                # Do not detect names below this limit, skip them
                'min_name_len': 1,
                }

        self.linking = {
                'learning_rate': 1,
                'anneal': True,
                # All concepts below this will always be disambiguated
                'disamb_length_limit': 4,
                # Context vector sizes that will be calculated and used for linking
                'context_vector_sizes': {'long': 18, 'medium': 9, 'short': 3},
                }


        # Some regex that we will need
        self.word_skipper = re.compile('^({})$'.format('|'.join(self.preprocessing['words_to_skip'])))
        # Very agressive punct checker
        self.punct_checker = re.compile(r'[^A-Za-z0-9]+')
