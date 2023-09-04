from datetime import datetime
from pydantic import BaseModel, Extra, ValidationError
from pydantic.dataclasses import Any, Callable, Dict, Optional, Union
from pydantic.fields import ModelField
from typing import List, Set, Tuple, cast
from multiprocessing import cpu_count
import logging
import jsonpickle
from functools import partial
import re

from medcat.utils.hasher import Hasher
from medcat.utils.matutils import intersect_nonempty_set


logger = logging.getLogger(__name__)


def weighted_average(step: int, factor: float) -> float:
    return max(0.1, 1 - (step ** 2 * factor))


def workers(workers_override: Optional[int] = None) -> int:
    return max(cpu_count() - 1, 1) if workers_override is None else workers_override


class FakeDict:
    """FakeDict that allows the use of the __getitem__ and __setitem__ method for legacy access."""

    def __getitem__(self, arg: str) -> Any:
        try:
            return getattr(self, arg)
        except AttributeError as e:
            raise KeyError from e


    def __setitem__(self, arg: str, val) -> None:
        setattr(self, arg, val)

    def get(self, key, default=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


_EMPTY_DICT_2_EMPTY_SET: Callable[[str, Any], Optional[set]] = lambda rhs, val: set() if (val == {} or rhs == "{}") else None


class ValueExtractor:
    """The current example config has a value for an empty set as '{}'.
    However, that evaluates to an empty dictionary instead.
    In case there are other such examples, this allows adding other alternatives as well.
    """

    def __init__(self, alt_generators: List[Callable[[str, Any], Optional[Any]]] = [_EMPTY_DICT_2_EMPTY_SET]) -> None:
        self.alt_generators = alt_generators

    def extract(self, rhs: str) -> Tuple[str, List[str]]:
        """Extracts value and its alternatives based on the alternative generators defined.

        Args:
            rhs(str): The parsable right hand side

        Returns:
            Tuple[str, List[str]]: The main value and the (potentially many) alternatives
        """
        val = eval(rhs)
        alts = []
        for gen in self.alt_generators:
            alt_val = gen(rhs, val)
            if alt_val is not None:
                alts.append(alt_val)
        return val, alts


_DEFAULT_EXTRACTOR = ValueExtractor()


def _set_value_or_alt(conf: 'MixingConfig', key: str, value: Any, alt_values: List[Any], err: Optional[ValidationError] = None) -> None:
    try:
        setattr(conf, key, value) # hoping for correct type
    except ValidationError as ve:
        if len(alt_values) > 0:
            _set_value_or_alt(conf, key, alt_values.pop(), alt_values, err=ve)
        elif err is not None:
            raise err
        else:
            raise ve


class MixingConfig(FakeDict):
    """Config that is able to saved and loaded from disk as well as mixed with other configs.
    It is not intended to be initialised directly and it is assumed that instances also inherit from
    pydantic's BaseModel.
    """

    def save(self, save_path: str) -> None:
        """Save the config into a .json file

        Args:
            save_path(str): Where to save the created json file
        """
        # We want to save the dict here, not the whole class
        json_string = jsonpickle.encode(
            {field: getattr(self, field) for field in self.fields()})

        with open(save_path, 'w') as f:
            f.write(json_string)

    def merge_config(self, config_dict: Dict) -> None:
        """Merge a config_dict with the existing config object.

        Args:
            config_dict(Dict): A dictionary which key/values should be added to this class.
        """
        for key in config_dict.keys():
            if hasattr(self, key):
                attr = getattr(self, key)
            else: # TODO - log this?
                attr = None # new attribute
            value = config_dict[key]
            if isinstance(value, BaseModel):
                value = value.dict()
            if isinstance(attr, MixingConfig):
                attr.merge_config(value)
            else:
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.warning('Issue with setting attribtue "%s":', key, exc_info=err)
        self.rebuild_re()

    def parse_config_file(self, path: str, extractor: ValueExtractor = _DEFAULT_EXTRACTOR) -> None:
        """Parses a configuration file in text format. Must be like:
                cat.<variable>.<key> = <value>
                ...

            - variable: linking, general, ner, ...
            - key: a key in the config dict e.g. subsample_after for linking
            - value: the value for the key, will be parsed with `eval`

        Args:
            path(str): the path to the config file
            extractor(ValueExtractor, optional):  (Default value = _DEFAULT_EXTRACTOR)
        """
        with open(path, 'r') as f:
            for line in f:
                if line.strip() and line.startswith("cat."):
                    line = line[4:]
                    left, right = line.split("=")
                    variable, key = left.split(".")
                    variable = variable.strip()
                    key = key.strip()
                    value, alt_values = extractor.extract(right)

                    attr = getattr(self, variable)
                    if isinstance(attr, MixingConfig):
                        _set_value_or_alt(attr, key, value, alt_values)
                    elif isinstance(attr, dict):
                        attr[key] = value
                    else:
                        raise ValueError(f'Unknown attribute {attr} for "{line}"')

        self.rebuild_re()

    def rebuild_re(self) -> None:
        pass

    def _calc_hash(self, hasher: Optional[Hasher] = None) -> Hasher:
        if hasher is None:
            hasher = Hasher()
        for _, v in cast(BaseModel, self).dict().items():
            if isinstance(v, MixingConfig):
                v._calc_hash(hasher)
            else:
                hasher.update(v)
        return hasher

    def get_hash(self, hasher: Optional[Hasher] = None):
        hasher = self._calc_hash(hasher)
        return hasher.hexdigest()

    def __str__(self) -> str:
        return str(cast(BaseModel, self).dict())

    @classmethod
    def load(cls, save_path: str) -> "MixingConfig":
        """Load config from a json file, note that fields that
        did not exist in the old config but do exist in the current
        version of the ConfigMetaCAT class will be kept.

        Args:
            save_path(str): Path to the json file to load

        Returns:
            MixingConfig: The loaded config
        """
        config = cls()

        # Read the jsonpickle string
        with open(save_path) as f:
            config_dict = jsonpickle.decode(f.read())

        config.merge_config(config_dict)

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MixingConfig":
        """Generate a MixingConfig (of an extending type) from a a dictionary.

        Args:
            config_dict(Dict): The dictionary to create the config from

        Returns:
            MixingConfig: The resulting config
        """
        config = cls()
        config.merge_config(config_dict)
        return config

    def asdict(self) -> Dict[str, Any]:
        """Get the config as a dictionary.

        Returns:
            Dict[str, Any]: The dictionary associated with this config
        """
        return cast(BaseModel, self).dict()

    def fields(self) -> Dict[str, ModelField]:
        """Get the fields associated with this config.

        Returns:
            Dict[str, Field]: The dictionary of the field names and fields
        """
        return cast(BaseModel, self).__fields__


class VersionInfo(MixingConfig, BaseModel):
    """The version info part of the config"""
    history: list = []
    """Populated automatically"""
    meta_cats: Any = {}
    """Populated automatically"""
    cdb_info: dict = {}
    """Populated automatically, output from cdb.print_stats"""
    performance: dict = {'ner': {}, 'meta': {}}
    """NER general performance, meta should be: {'meta': {'model_name': {'f1': <>, 'p': <>, ...}, ...}}"""
    description: str = "No description"
    """General description and what it was trained on"""
    id: Any = None
    """Will be: hash of most things"""
    last_modified: Optional[Union[int, datetime, str]] = None
    location: Optional[str] = None
    """Path/URL/Whatever to where is this CDB located"""
    ontology: Optional[Union[str, List[str]]] = None
    """What was used to build the CDB, e.g. SNOMED_202009"""
    medcat_version: Optional[str] = None
    """Which version of medcat was used to build the CDB"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class CDBMaker(MixingConfig, BaseModel):
    """The Context Database (CDB) making part of the config"""
    name_versions: list = ['LOWER', 'CLEAN']
    """Name versions to be generated."""
    multi_separator: str = '|'
    """If multiple names or type_ids for a concept present in one row of a CSV, they are separted
    by the character below."""
    remove_parenthesis: int = 5
    """Should preferred names with parenthesis be cleaned 0 means no, else it means if longer than or equal
    e.g. Head (Body part) -> Head"""
    min_letters_required: int = 2
    """Minimum number of letters required in a name to be accepted for a concept"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class AnnotationOutput(MixingConfig, BaseModel):
    """The annotation output part of the config"""
    doc_extended_info: bool = False
    context_left: int = -1
    context_right: int = -1
    lowercase_context: bool = True
    include_text_in_output: bool = False

    class Config:
        extra = Extra.allow
        validate_assignment = True


class CheckPoint(MixingConfig, BaseModel):
    """The checkpoint part of the config"""
    output_dir: str = 'checkpoints'
    """When doing training this is the name of the directory where checkpoints will be saved"""
    steps: Optional[int] = None
    """When training how often to save the checkpoint (one step represents one document), if None no ckpts will be created"""
    max_to_keep: int = 1
    """When training the maximum checkpoints will be kept on the disk"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class General(MixingConfig, BaseModel):
    """The general part of the config"""
    spacy_disabled_components: list = ['ner', 'parser', 'vectors', 'textcat',
                                       'entity_linker', 'sentencizer', 'entity_ruler', 'merge_noun_chunks',
                                       'merge_entities', 'merge_subtokens']
    checkpoint: CheckPoint = CheckPoint()
    """Checkpointing config"""
    log_level: int = logging.INFO
    """Logging config for everything | 'tagger' can be disabled, but will cause a drop in performance"""
    log_format: str = '%(levelname)s:%(name)s: %(message)s'
    log_path: str = './medcat.log'
    spacy_model: str = 'en_core_web_md'
    """What model will be used for tokenization"""
    separator: str = '~'
    """Separator that will be used to merge tokens of a name. Once a CDB is built this should
    always stay the same."""
    spell_check: bool = True
    """Should we check spelling - note that this makes things much slower, use only if necessary. The only thing necessary
    for the spell checker to work is vocab.dat and cdb.dat built with concepts in the respective language."""
    diacritics: bool = False
    """Should we process diacritics - for languages other than English, symbols such as 'é, ë, ö' can be relevant.
    Note that this makes spell_check slower."""
    spell_check_deep: bool = False
    """If True the spell checker will try harder to find mistakes, this can slow down
    things drastically."""
    spell_check_len_limit: int = 7
    """Spelling will not be checked for words with length less than this"""
    show_nested_entities: bool = False
    """If set to True functions like get_entities and get_json will return nested_entities and overlaps"""
    full_unlink: bool = False
    """When unlinking a name from a concept should we do full_unlink (means unlink a name from all concepts, not just the one in question)"""
    workers: int = workers()
    """Number of workers used by a parallelizable pipeline component"""
    make_pretty_labels: Optional[str] = None
    """Should the labels of entities (shown in displacy) be pretty or just 'concept'. Slows down the annotation pipeline
    should not be used when annotating millions of documents. If `None` it will be the string "concept", if `short` it will be CUI,
    if `long` it will be CUI | Name | Confidence"""
    map_cui_to_group: bool = False
    """If the cdb.addl_info['cui2group'] is provided and this option enabled, each CUI will be maped to the group"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class Preprocessing(MixingConfig, BaseModel):
    """The preprocessing part of the config"""
    words_to_skip: set = {'nos'}
    """This words will be completly ignored from concepts and from the text (must be a Set)"""
    keep_punct: set = {'.', ':'}
    """All punct will be skipped by default, here you can set what will be kept"""
    do_not_normalize: set = {'VBD', 'VBG', 'VBN', 'VBP', 'JJS', 'JJR'}
    """Should specific word types be normalized: e.g. running -> run
    Values are detailed part-of-speech tags. See:
    - https://spacy.io/usage/linguistic-features#pos-tagging
    - Label scheme section per model at https://spacy.io/models/en"""
    skip_stopwords: bool = False
    """Should stopwords be skipped/ingored when processing input"""
    min_len_normalize: int = 5
    """Nothing below this length will ever be normalized (input tokens or concept names), normalized means lemmatized in this case"""
    stopwords: Optional[set] = None
    """If None the default set of stowords from spacy will be used. This must be a Set."""
    max_document_length: int = 1000000
    """Documents longer  than this will be trimmed"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class Ner(MixingConfig, BaseModel):
    """The NER part of the config"""
    min_name_len: int = 3
    """Do not detect names below this limit, skip them"""
    max_skip_tokens: int = 2
    """When checkng tokens for concepts you can have skipped tokens inbetween
    used ones (usually spaces, new lines etc). This number tells you how many skipped can you have."""
    check_upper_case_names: bool = False
    """Check uppercase to distinguish uppercase and lowercase words that have a different meaning."""
    upper_case_limit_len: int = 4
    """Any name shorter than this must be uppercase in the text to be considered. If it is not uppercase
    it will be skipped."""
    try_reverse_word_order: bool = False
    """Try reverse word order for short concepts (2 words max), e.g. heart disease -> disease heart"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class _DefPartial:
    """This is a helper class to make it possible to check equality of two default Linking instances"""

    def __init__(self):
        self.fun = partial(weighted_average, factor=0.0004)

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

    def __eq__(self, other):
        return isinstance(other, _DefPartial)


_DEFAULT_PARTIAL = _DefPartial()


class LinkingFilters(MixingConfig, BaseModel):
    """These describe the linking filters used alongside the model.

    When no CUIs nor exlcuded CUIs are specified (the sets are empty),
    all CUIs are accepted.
    If there are CUIs specified then only those will be accepted.
    If there are excluded CUIs specified, they are excluded.

    In some cases, there are extra filters as well as MedCATtrainer (MCT) export filters.
    These are expcted to follow the following:
    extra_cui_filter ⊆ MCT filter ⊆ Model/config filter

    While any other CUIs can be included in the the extra CUI filter or the MCT filter,
    they would not have any real effect.
    """
    cuis: Set[str] = set()
    cuis_exclude: Set[str] = set()

    def check_filters(self, cui: str) -> bool:
        """Checks is a CUI in the filters

        Args:
            cui (str): The CUI in question

        Returns:
            bool: True if the CUI is allowed
        """
        if cui in self.cuis or not self.cuis:
            return cui not in self.cuis_exclude
        else:
            return False

    def merge_with(self, other: 'LinkingFilters') -> None:
        """Merge CUIs and excluded CUIs within two filters.
        The data will be kept within this filter (and not the other).

        Args:
            other (LinkingFilters): The other filter to merge with
        """
        self.cuis = intersect_nonempty_set(other.cuis, self.cuis)
        self.cuis_exclude.update(other.cuis_exclude) # TODO - something different?

    def copy_of(self) -> 'LinkingFilters':
        """Create a copy of this LinkingFilters.
        This copy will describe an identical filter but will refer to
        different sets so they can be mutated separately.

        Returns:
            LinkingFilters: A copy of the original filters.
        """
        return LinkingFilters(cuis=set(self.cuis), cuis_exclude=set(self.cuis_exclude))


class Linking(MixingConfig, BaseModel):
    """The linking part of the config"""
    optim: dict = {'type': 'linear', 'base_lr': 1, 'min_lr': 0.00005}
    """Linear anneal"""
    # optim: dict = {'type': 'standard', 'lr': 1}
    # optim: dict = {'type': 'moving_avg', 'alpha': 0.99, 'e': 1e-4, 'size': 100}
    context_vector_sizes: dict = {'xlong': 27, 'long': 18, 'medium': 9, 'short': 3}
    """Context vector sizes that will be calculated and used for linking"""
    context_vector_weights: dict = {'xlong': 0.1, 'long': 0.4, 'medium': 0.4, 'short': 0.1}
    """Weight of each vector in the similarity score - make trainable at some point. Should add up to 1."""
    filters: LinkingFilters = LinkingFilters()
    """Filters"""
    train: bool = True
    """Should it train or not, this is set automatically ignore in 99% of cases and do not set manually"""
    random_replacement_unsupervised: float = 0.80
    """If <1 during unsupervised training the detected term will be randomly replaced with a probability of 1 - random_replacement_unsupervised
    Replaced with a synonym used for that term"""
    disamb_length_limit: int = 3
    """All concepts below this will always be disambiguated"""
    filter_before_disamb: bool = False
    """If True it will filter before doing disamb. Useful for the trainer."""
    train_count_threshold: int = 1
    """Concepts that have seen less training examples than this will not be used for
    similarity calculation and will have a similarity of -1."""
    always_calculate_similarity: bool = False
    """Do we want to calculate context similarity even for concepts that are not ambigous."""
    weighted_average_function: Callable[..., Any] = _DEFAULT_PARTIAL
    """Weights for a weighted average
    'weighted_average_function': partial(weighted_average, factor=0.02),"""
    calculate_dynamic_threshold: bool = False
    """Concepts below this similarity will be ignored. Type can be static/dynamic - if dynamic each CUI has a different TH
    and it is calcualted as the average confidence for that CUI * similarity_threshold. Take care that dynamic works only
    if the cdb was trained with calculate_dynamic_threshold = True."""
    similarity_threshold_type: str = 'static'
    similarity_threshold: float = 0.25
    negative_probability: float = 0.5
    """Probability for the negative context to be added for each positive addition"""
    negative_ignore_punct_and_num: bool = True
    """Do we ignore punct/num when negative sampling"""
    prefer_primary_name: float = 0.35
    """If >0 concepts for which a detection is its primary name will be preferred by that amount (0 to 1)"""
    prefer_frequent_concepts: float = 0.35
    """If >0 concepts that are more frequent will be prefered by a multiply of this amount"""
    subsample_after: int = 30000
    """DISABLED in code permanetly: Subsample during unsupervised training if a concept has received more than"""
    devalue_linked_concepts: bool = False
    """When adding a positive example, should it also be treated as Negative for concepts
    which link to the postive one via names (ambigous names)."""
    context_ignore_center_tokens: bool = False
    """If true when the context of a concept is calculated (embedding) the words making that concept are not taken into accout"""

    class Config:
        extra = Extra.allow
        validate_assignment = True


class Config(MixingConfig, BaseModel):
    """The MedCAT config"""
    version: VersionInfo = VersionInfo()
    cdb_maker: CDBMaker = CDBMaker()
    annotation_output: AnnotationOutput = AnnotationOutput()
    general: General = General()
    preprocessing: Preprocessing = Preprocessing()
    ner: Ner = Ner()
    linking: Linking = Linking()
    word_skipper: re.Pattern = re.compile('') # empty pattern gets replaced upon init
    punct_checker: re.Pattern = re.compile('') # empty pattern gets replaced upon init

    class Config:
        # this if for word_skipper and punct_checker which would otherwise
        # not have a validator
        arbitrary_types_allowed = True
        extra = Extra.allow
        validate_assignment = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rebuild_re()

    # Override
    def rebuild_re(self) -> None:
        # Some regex that we will need
        self.word_skipper = re.compile('^({})$'.format(
            '|'.join(self.preprocessing.words_to_skip)))
        # Very agressive punct checker, input will be lowercased
        self.punct_checker = re.compile(r'[^a-z0-9]+')

    # Override
    def get_hash(self):
        hasher = Hasher()
        for k, v in self.dict().items():
            if k not in ['version', 'general', 'linking']:
                hasher.update(v, length=True)
            elif k == 'general':
                for k2, v2 in v.items():
                    if k2 != 'spacy_model':
                        hasher.update(v2, length=False)
                    else:
                        # Ignore spacy model
                        pass
            elif k == 'linking':
                for k2, v2 in v.items():
                    if k2 != "filters":
                        hasher.update(v2, length=False)
                    else:
                        hasher.update(v2, length=True)

        return hasher.hexdigest()
