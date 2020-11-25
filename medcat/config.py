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
                # Logging config for everything
                'log_level': logging.DEBUG,
                'log_format': '%(asctime)s: %(message)s',
                'spacy_disabled_components': ['tagger', 'ner', 'parser', 'vectors', 'textcat',
                                              'entity_linker', 'sentencizer', 'entity_ruler', 'merge_noun_chunks',
                                              'merge_entities', 'merge_subtokens'],
                # What model will be used for tokenization
                'spacy_model': 'en_core_sci_md',
                # Is MedCATtrainer running MedCAT
                'is_trainer': False,
                # Should we check spelling - note that this makes things much slower, use only if necessary. The only thing necessary
                #for the spell checker to work is vocab.dat and cdb.dat built with concepts in the respective language.
                'spell_check': True,
                # Spelling will not be checked for words with less than the limit below
                'spell_check_len_limit': 5,
                }

        self.preprocessing = {
                # Should stopwords be skipped/ingored when processing input
                'skip_stopwords': False,
                # This words will be completly ignored from concepts and from the text (must be a Set)
                'words_to_skip': set(['and', 'or', 'nos']),
                # All punct will be skipped by default, here you can set what will be kept
                'keep_punct': {'.', ':'},
                # Nothing below this length will ever be normalized (input tokens or concept names), normalized means lemmatized in this case
                'min_len_normalize': 5,
                # If None the default set of stowords from spacy will be used. This must be a Set.
                'stopwords': None,
                }

        self.ner = {
                # Do not detect names below this limit, skip them
                'min_name_len': 1,
                # When checkng tokens for concepts you can have skipped tokens inbetween
                #used ones (usually spaces, new lines etc). This number tells you how many skipped can you have.
                'max_skip_tokens': 2,
                # Any name shorter than this must be uppercase in the text to be considered.
                'upper_case_limit_len': 4,
                }

        self.linking = {
                # Linear anneal
                'optim': {'type': 'linear', 'base_lr': 1, 'min_lr': 0.00005},
                # 'optim': {'standard': 'lr': 1},
                # 'optim': {'moving_avg': 'alpha': 0.99, 'e': 1e-4, 'size': 100},
                # Useful only if we have anneal
                'min_learning_rate': 1e-5,
                # All concepts below this will always be disambiguated
                'disamb_length_limit': 4,
                # Context vector sizes that will be calculated and used for linking
                'context_vector_sizes': {'long': 18, 'medium': 9, 'short': 3},
                # Weight of each vector in the similarity score - make trainable at some point. Should add up to 1.
                'context_vector_weights': {'long': 0.2, 'medium': 0.6, 'short': 0.2},
                # Do we prefer frequent concepts over others
                'prefer_frequent_concepts': False,
                # Concepts with this tag/tags will be prefered
                'prefer_concepts_with_tag': [], # e.g. ['my_super_tag', 'and_one_more']

                'prefer_concepts_with_primary_name': 1,
                'devalue_short_names': 1,

                # Concepts that have seen less training examples than this will not be used for
                #similarity calculation and will have a similarity of -1.
                'train_count_threshold': 20,
                # Do we want to calculate context similarity even for concepts that are not ambigous.
                'always_calculate_similarity': False,
                # Weights for a weighted average
                'weighted_average_function': lambda step: max(0.1, 1-(step**2*0.02)),
                # Concepts below this similarity will be ignored
                'similarity_threshold': 0.2,
                # Probability for the negative context to be added for each positive addition
                'negative_probability': 1,
                # Do we ignore punct/num when negative sampling
                'negative_ignore_punct_and_num': True,

                # REMOVE
                'filters': {
                    'exclude': set(), # CUIs in this filter will be excluded, everything else included - will be ignored if 'include'
                    'include': set(), # CUIs in this filter will be included, everything else excluded
                    }
                }


        # Some regex that we will need
        self.word_skipper = re.compile('^({})$'.format('|'.join(self.preprocessing['words_to_skip'])))
        # Very agressive punct checker, input will be lowercased
        self.punct_checker = re.compile(r'[^a-z0-9]+')
