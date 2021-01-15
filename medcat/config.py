import re
import logging

class Config(object):
    def __init__(self):
        # CDB Maker
        self.cdb_maker = {
               # If multiple names or type_ids for a concept present in one row of a CSV, they are separted
                #by the character below.
                'multi_separator': '|',
                # Name versions to be generated.
                'name_versions': ['CLEAN', 'LOWER'],
                }

        self.general = {
                # Logging config for everything
                'log_level': logging.INFO,
                'log_format': '%(asctime)s: %(message)s',
                'spacy_disabled_components': ['tagger', 'ner', 'parser', 'vectors', 'textcat',
                                              'entity_linker', 'sentencizer', 'entity_ruler', 'merge_noun_chunks',
                                              'merge_entities', 'merge_subtokens'],
                # What model will be used for tokenization
                'spacy_model': 'en_core_sci_lg',
                # Separator that will be used to merge tokens of a name. Once a CDB is built this should
                #always stay the same.
                'separator': '~',
                # Should we check spelling - note that this makes things much slower, use only if necessary. The only thing necessary
                #for the spell checker to work is vocab.dat and cdb.dat built with concepts in the respective language.
                'spell_check': False,
                # If True the spell checker will try harder to find mistakes, this can slow down
                #things drastically.
                'spell_check_deep': False,
                # Spelling will not be checked for words with length less than this
                'spell_check_len_limit': 7,
                # If set to True functions like get_entities and get_json will return nested_entities and overlaps
                'show_nested_entities': False,
                # When unlinking a name from a concept should we do full_unlink
                'full_unlink': False,
                }

        self.preprocessing = {
                # Should stopwords be skipped/ingored when processing input
                'skip_stopwords': False,
                # This words will be completly ignored from concepts and from the text (must be a Set)
                'words_to_skip': set(['nos']),
                # All punct will be skipped by default, here you can set what will be kept
                'keep_punct': {'.', ':'},
                # Nothing below this length will ever be normalized (input tokens or concept names), normalized means lemmatized in this case
                'min_len_normalize': 5,
                # If None the default set of stowords from spacy will be used. This must be a Set.
                'stopwords': None,
                }

        self.ner = {
                # Do not detect names below this limit, skip them
                'min_name_len': 3,
                # When checkng tokens for concepts you can have skipped tokens inbetween
                #used ones (usually spaces, new lines etc). This number tells you how many skipped can you have.
                'max_skip_tokens': 2,
                # Any name shorter than this must be uppercase in the text to be considered. If it is not uppercase
                #it will be skipped.
                'upper_case_limit_len': 4,
                }

        self.linking = {
                # Should it train or not, this is set automatically ignore in 99% of cases and do not set manually
                'train': True,
                # Linear anneal
                'optim': {'type': 'linear', 'base_lr': 1, 'min_lr': 0.00005},
                # 'optim': {'type': 'standard', 'lr': 1},
                # 'optim': {'type': 'moving_avg', 'alpha': 0.99, 'e': 1e-4, 'size': 100},
                # All concepts below this will always be disambiguated
                'disamb_length_limit': 5,
                # Context vector sizes that will be calculated and used for linking
                'context_vector_sizes': {'xxxlong': 60, 'xxlong': 45, 'xlong': 27, 'long': 18, 'medium': 9, 'short': 3},
                # Weight of each vector in the similarity score - make trainable at some point. Should add up to 1.
                'context_vector_weights': {'xxxlong': 0, 'xxlong': 0, 'xlong': 0, 'long': 0.4, 'medium': 0.4, 'short': 0.2},
                # If True it will filter before doing disamb. Useful for the trainer.
                'filter_before_disamb': False,
                # Concepts that have seen less training examples than this will not be used for
                #similarity calculation and will have a similarity of -1.
                'train_count_threshold': 2,
                # Do we want to calculate context similarity even for concepts that are not ambigous.
                'always_calculate_similarity': False,
                # Weights for a weighted average
                'weighted_average_function': lambda step: max(0.1, 1-(step**2*0.02)),
                # Concepts below this similarity will be ignored
                'similarity_threshold': 0.2,
                # Probability for the negative context to be added for each positive addition
                'negative_probability': 0.5,
                # Do we ignore punct/num when negative sampling
                'negative_ignore_punct_and_num': True,
                # If True concepts for which a detection is its primary name will be preferred
                'prefer_primary_name': True,
                # Subsample during unsupervised training if a concept has received more than 
                'subsample_after': 30000,
                # Filters
                'filters': {
                    'cuis': set(), # CUIs in this filter will be included, everything else excluded, must be a set, if empty all cuis will be included
                    }
                }


        # Some regex that we will need
        self.word_skipper = re.compile('^({})$'.format('|'.join(self.preprocessing['words_to_skip'])))
        # Very agressive punct checker, input will be lowercased
        self.punct_checker = re.compile(r'[^a-z0-9]+')


    @classmethod
    def from_dict(cls, d):
        config = cls()
        config.__dict__ = d

        return config

    def rebuild_re(self):
        # Some regex that we will need
        self.word_skipper = re.compile('^({})$'.format('|'.join(self.preprocessing['words_to_skip'])))
        # Very agressive punct checker, input will be lowercased
        self.punct_checker = re.compile(r'[^a-z0-9]+')


    def parse_config_file(self, path):
        r'''
        Parses a configuration file in text format. Must be like:
                <variable>.<key> = <value>
                ...
            Where:
                variable: linking, general, ner, ...
                key: a key in the config dict e.g. subsample_after for linking
                value: the value for the key, will be parsed with `eval`
        '''
        with open(path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    left, right = line.split("=")
                    variable, key = left.split(".")
                    variable = variable.strip()
                    key = key.strip()
                    value = eval(right.strip())

                    attr = getattr(self, variable)
                    attr[key] = value
