import re
import logging
import jsonpickle
from functools import partial
from multiprocessing import cpu_count


def weighted_average(step, factor):
    return max(0.1, 1 - (step ** 2 * factor))


def workers(workers_override=None):
    return max(cpu_count() - 1, 1) if workers_override is None else workers_override


class BaseConfig(object):
    jsonpickle.set_encoder_options('json', sort_keys=True, indent=2)


    def __init__(self):
        pass


    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


    def save(self, save_path):
        r''' Save the config into a .json file

        Args:
            save_path (`str`):
                Where to save the created json file
        '''
        # We want to save the dict here, not the whole class
        json_string = jsonpickle.encode(self.__dict__)

        with open(save_path, 'w') as f:
            f.write(json_string)


    def merge_config(self, config_dict):
        r''' Merge a config_dict with the existing config object.

        Args:
            config_dict (`dict`):
                A dictionary which key/values should be added to this class.
        '''
        for key in config_dict.keys():
            if key in self.__dict__ and isinstance(self.__dict__[key], dict):
                self.__dict__[key].update(config_dict[key])
            else:
                self.__dict__[key] = config_dict[key]


    def parse_config_file(self, path):
        r'''
        Parses a configuration file in text format. Must be like:
                cat.<variable>.<key> = <value>
                ...
            Where:
                variable: linking, general, ner, ...
                key: a key in the config dict e.g. subsample_after for linking
                value: the value for the key, will be parsed with `eval`
        '''
        with open(path, 'r') as f:
            for line in f:
                if line.strip() and line.startswith("cat."):
                    line = line[4:]
                    left, right = line.split("=")
                    variable, key = left.split(".")
                    variable = variable.strip()
                    key = key.strip()
                    value = eval(right.strip())

                    attr = getattr(self, variable)
                    attr[key] = value

        self.rebuild_re()


    def rebuild_re():
        pass


    def __str__(self):
        json_obj = {}
        for attr, value in self:
            json_obj[attr] = value
        return jsonpickle.encode(json_obj)


    @classmethod
    def load(cls, save_path):
        r''' Load config from a json file, note that fields that
        did not exist in the old config but do exist in the current
        version of the ConfigMetaCAT class will be kept.

        Args:
            save_path (`str`):
                Path to the json file to load
        '''
        config = cls()

        # Read the jsonpickle string
        with open(save_path) as f:
            config_dict = jsonpickle.decode(f.read())

        config.merge_config(config_dict)

        return config


    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        config.merge_config(config_dict)

        return config


class Config(BaseConfig):

    jsonpickle.set_encoder_options('json', sort_keys=True, indent=2)

    def __init__(self):
        super().__init__()

        # CDB Maker
        self.cdb_maker = {
                # If multiple names or type_ids for a concept present in one row of a CSV, they are separted
                # by the character below.
                'multi_separator': '|',
                # Name versions to be generated.
                'name_versions': ['LOWER', 'CLEAN'],
                # Should preferred names with parenthesis be cleaned 0 means no, else it means if longer than or equal
                # e.g. Head (Body part) -> Head
                'remove_parenthesis': 5,
                # Minimum number of letters required in a name to be accepted for a concept
                'min_letters_required': 2,
                }

        # Used mainly to configure the output of the get_entities function, and in that also the output of
        #get_json and multiprocessing
        self.annotation_output = {
                'doc_extended_info': False,
                'context_left': -1,
                'context_right': -1,
                'lowercase_context': True,
                'include_text_in_output': False,
                }

        self.general = {
                # What was used to build the CDB, e.g. SNOMED_202009
                'cdb_source_name': '',
                # Logging config for everything | 'tagger' can be disabled, but will cause a drop in performance
                'log_level': logging.INFO,
                'log_format': '%(levelname)s:%(name)s: %(message)s',
                'log_path': './medcat.log',
                'spacy_disabled_components': ['ner', 'parser', 'vectors', 'textcat',
                                              'entity_linker', 'sentencizer', 'entity_ruler', 'merge_noun_chunks',
                                              'merge_entities', 'merge_subtokens'],
                # What model will be used for tokenization
                'spacy_model': 'en_core_web_md',
                # Separator that will be used to merge tokens of a name. Once a CDB is built this should
                #always stay the same.
                'separator': '~',
                # Should we check spelling - note that this makes things much slower, use only if necessary. The only thing necessary
                #for the spell checker to work is vocab.dat and cdb.dat built with concepts in the respective language.
                'spell_check': True,
                # Should we process diacritics - for languages other than English, symbols such as 'é, ë, ö' can be relevant.
                # Note that this makes spell_check slower.
                'diacritics': False,
                # If True the spell checker will try harder to find mistakes, this can slow down
                #things drastically.
                'spell_check_deep': False,
                # Spelling will not be checked for words with length less than this
                'spell_check_len_limit': 7,
                # If set to True functions like get_entities and get_json will return nested_entities and overlaps
                'show_nested_entities': False,
                # When unlinking a name from a concept should we do full_unlink (means unlink a name from all concepts, not just the one in question)
                'full_unlink': False,
                # Number of workers used by a parallelizable pipeline component
                'workers': workers(),
                # Should the labels of entities (shown in displacy) be pretty or just 'concept'. Slows down the annotation pipeline
                #should not be used when annotating millions of documents. If `None` it will be the string "concept", if `short` it will be CUI,
                #if `long` it will be CUI | Name | Confidence
                'make_pretty_labels': None,
                # If the cdb.addl_info['cui2group'] is provided and this option enabled, each CUI will be maped to the group
                'map_cui_to_group': False,
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
                # Documents longer  than this will be trimmed
                'max_document_length': 1000000,
                # Should specific word types be normalized: e.g. running -> run
                'do_not_normalize': {'VBD', 'VBG', 'VBN', 'VBP', 'JJS', 'JJR'},
                }

        self.ner = {
                # Do not detect names below this limit, skip them
                'min_name_len': 3,
                # When checkng tokens for concepts you can have skipped tokens inbetween
                #used ones (usually spaces, new lines etc). This number tells you how many skipped can you have.
                'max_skip_tokens': 2,
                # Check uppercase to distinguish uppercase and lowercase words that have a different meaning.
                'check_upper_case_names': False,
                # Any name shorter than this must be uppercase in the text to be considered. If it is not uppercase
                #it will be skipped.
                'upper_case_limit_len': 3,
                # Try reverse word order for short concepts (2 words max), e.g. heart disease -> disease heart
                'try_reverse_word_order': False,
                }

        self.linking = {
                # Should it train or not, this is set automatically ignore in 99% of cases and do not set manually
                'train': True,
                # Linear anneal
                'optim': {'type': 'linear', 'base_lr': 1, 'min_lr': 0.00005},
                # 'optim': {'type': 'standard', 'lr': 1},
                # 'optim': {'type': 'moving_avg', 'alpha': 0.99, 'e': 1e-4, 'size': 100},
                # All concepts below this will always be disambiguated
                'disamb_length_limit': 3,
                # Context vector sizes that will be calculated and used for linking
                'context_vector_sizes': {'xlong': 27, 'long': 18, 'medium': 9, 'short': 3},
                # Weight of each vector in the similarity score - make trainable at some point. Should add up to 1.
                'context_vector_weights': {'xlong': 0.1, 'long': 0.4, 'medium': 0.4, 'short': 0.1},
                # If True it will filter before doing disamb. Useful for the trainer.
                'filter_before_disamb': False,
                # Concepts that have seen less training examples than this will not be used for
                #similarity calculation and will have a similarity of -1.
                'train_count_threshold': 1,
                # Do we want to calculate context similarity even for concepts that are not ambigous.
                'always_calculate_similarity': False,
                # Weights for a weighted average
                #'weighted_average_function': partial(weighted_average, factor=0.02),
                'weighted_average_function': partial(weighted_average, factor=0.0004),
                # Concepts below this similarity will be ignored. Type can be static/dynamic - if dynamic each CUI has a different TH
                #and it is calcualted as the average confidence for that CUI * similarity_threshold. Take care that dynamic works only
                #if the cdb was trained with calculate_dynamic_threshold = True.
                'calculate_dynamic_threshold': False,
                'similarity_threshold_type': 'static',
                'similarity_threshold': 0.2,
                # Probability for the negative context to be added for each positive addition
                'negative_probability': 0.5,
                # Do we ignore punct/num when negative sampling
                'negative_ignore_punct_and_num': True,
                # If >0 concepts for which a detection is its primary name will be preferred by that amount (0 to 1)
                'prefer_primary_name': 0.35,
                # If >0 concepts that are more frequent will be prefered by a multiply of this amount
                'prefer_frequent_concepts': 0.35,
                # Subsample during unsupervised training if a concept has received more than
                'subsample_after': 30000,
                # When adding a positive example, should it also be treated as Negative for concepts
                #which link to the postive one via names (ambigous names).
                'devalue_linked_concepts': False,
                # If true when the context of a concept is calculated (embedding) the words making that concept are not taken into accout
                'context_ignore_center_tokens': False,
                # Filters
                'filters': {
                    'cuis': set(), # CUIs in this filter will be included, everything else excluded, must be a set, if empty all cuis will be included
                    },
                }


        # Some regex that we will need
        self.word_skipper = re.compile('^({})$'.format('|'.join(self.preprocessing['words_to_skip'])))
        # Very agressive punct checker, input will be lowercased
        self.punct_checker = re.compile(r'[^a-z0-9]+')


    def rebuild_re(self):
        # Some regex that we will need
        self.word_skipper = re.compile('^({})$'.format('|'.join(self.preprocessing['words_to_skip'])))
        # Very agressive punct checker, input will be lowercased
        self.punct_checker = re.compile(r'[^a-z0-9]+')
