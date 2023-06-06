import types
import spacy
import gc
import logging
from typing import List, Optional, Union, Iterable, Callable
from multiprocessing import cpu_count
from spacy.tokens import Token, Doc, Span
from spacy.tokenizer import Tokenizer
from spacy.language import Language
from spacy.util import raise_error
from tqdm.autonotebook import tqdm
from medcat.linking.context_based_linker import Linker
from medcat.meta_cat import MetaCAT
from medcat.ner.vocab_based_ner import NER
from medcat.utils.normalizers import TokenNormalizer, BasicSpellChecker
from medcat.config import Config
from medcat.pipeline.pipe_runner import PipeRunner
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.ner.transformers_ner import TransformersNER


logger = logging.getLogger(__name__) # different logger from the package-level one


class Pipe(object):
    """A wrapper around the standard spacy pipeline.

    Args:
        tokenizer (spacy.tokenizer.Tokenizer):
            What will be used to split text into tokens,
            can be anything built as a spacy tokenizer.
        config (medcat.config.Config):
            Global config for medcat.

    Properties:
        nlp (spacy.language.<lng>):
            The base spacy NLP pipeline.
    """

    def __init__(self, tokenizer: Tokenizer, config: Config) -> None:
        self._nlp = spacy.load(config.general.spacy_model, disable=config.general.spacy_disabled_components)
        if config.preprocessing.stopwords is not None:
            self._nlp.Defaults.stop_words = set(config.preprocessing.stopwords)
        self._nlp.tokenizer = tokenizer(self._nlp, config)
        # Set max document length
        self._nlp.max_length = config.preprocessing.max_document_length
        self.config = config
        # Set log level
        logger.setLevel(self.config.general.log_level)

    def add_tagger(self, tagger: Callable, name: Optional[str] = None, additional_fields: List[str] = []) -> None:
        """Add any kind of a tagger for tokens.

        Args:
            tagger(Callable):
                Any object/function that takes a spacy doc as an input, does something
                and returns the same doc.
            name(Optional[str], optional):
                Name for this component in the pipeline. (Default value = None)
            additional_fields(List[str], optional):
                Fields to be added to the `_` properties of a token. (Default value = [])
        """
        component_factory_name = spacy.util.get_object_name(tagger)
        name = name if name is not None else component_factory_name
        Language.factory(name=component_factory_name, default_config={"config": self.config}, func=tagger)
        self._nlp.add_pipe(component_factory_name, name=name, first=True)

        # Add custom fields needed for this usecase
        Token.set_extension('to_skip', default=False, force=True)

        # Add any additional fields that are required
        for field in additional_fields:
            Token.set_extension(field, default=False, force=True)

    def add_token_normalizer(self, config: Config, name: Optional[str] = None, spell_checker: Optional[BasicSpellChecker] = None) -> None:
        token_normalizer = TokenNormalizer(config=config, spell_checker=spell_checker)
        component_name = spacy.util.get_object_name(token_normalizer)
        name = name if name is not None else component_name
        Language.component(name=component_name, func=token_normalizer)
        self._nlp.add_pipe(component_name, name=name, last=True)

        # Add custom fields needed for this usecase
        Token.set_extension('norm', default=None, force=True)

    def add_ner(self, ner: NER, name: Optional[str] = None) -> None:
        """Add NER from CAT to the pipeline, will also add the necessary fields
        to the document and Span objects.

        Args:
            ner(NER):
                The NER instance
            name(Optional[str], optional):
                The pipeline name (Default value = None)
        """
        component_name = spacy.util.get_object_name(ner)
        name = name if name is not None else component_name
        Language.component(name=component_name, func=ner)
        self._nlp.add_pipe(component_name, name=name, last=True)

        Doc.set_extension('ents', default=[], force=True)
        Span.set_extension('confidence', default=-1, force=True)
        Span.set_extension('id', default=0, force=True)

        # Do not set this property if a vocabulary apporach is not used, this name must
        #refer to a name2cuis in the cdb.
        Span.set_extension('detected_name', default=None, force=True)
        Span.set_extension('link_candidates', default=None, force=True)

    def add_linker(self, linker: Linker, name: Optional[str] = None) -> None:
        """Add entity linker to the pipeline, will also add the necessary fields
        to Span object.

        Args:
            linker(Linker):
                Any object/function created based on the requirements for a spaCy pipeline components. Have
                a look at https://spacy.io/usage/processing-pipelines#custom-components
            name(Optional[str], optional):
                The component name (Default value = None)
        """
        component_name = spacy.util.get_object_name(linker)
        name = name if name is not None else component_name
        Language.component(name=component_name, func=linker)
        self._nlp.add_pipe(component_name, name=name, last=True)
        Span.set_extension('cui', default=-1, force=True)
        Span.set_extension('context_similarity', default=-1, force=True)

    def add_meta_cat(self, meta_cat: MetaCAT, name: Optional[str] = None) -> None:
        component_name = spacy.util.get_object_name(meta_cat)
        name = name if name is not None else component_name
        Language.component(name=component_name, func=meta_cat)
        self._nlp.add_pipe(component_name, name=name, last=True)

        # meta_anns is a dictionary like {category_name: value, ...}
        Span.set_extension('meta_anns', default=None, force=True)
        # Used for sharing pre-processed data/tokens
        Doc.set_extension('share_tokens', default=None, force=True)


    def add_addl_ner(self, addl_ner: TransformersNER, name: Optional[str] = None) -> None:
        component_name = spacy.util.get_object_name(addl_ner)
        name = name if name is not None else component_name
        Language.component(name=component_name, func=addl_ner)  # type: ignore
        self._nlp.add_pipe(component_name, name=name, last=True)

        Doc.set_extension('ents', default=[], force=True)
        Span.set_extension('confidence', default=-1, force=True)
        Span.set_extension('id', default=0, force=True)
        Span.set_extension('cui', default=-1, force=True)
        Span.set_extension('context_similarity', default=-1, force=True)
        Span.set_extension('detected_name', default=None, force=True)


    def batch_multi_process(self,
                            texts: Iterable[str],
                            n_process: Optional[int] = None,
                            batch_size: Optional[int] = None) -> Iterable[Doc]:
        """Batch process a list of texts in parallel.

        Args:
            texts (Iterable[str]):
                The input sequence of texts to process.
            n_process (int):
                The number of processes running in parallel.
                Defaults to max(mp.cpu_count() - 1, 1).
            batch_size (int):
                The number of texts to buffer. Defaults to 1000.
            total (int):
                The number of texts in total.

        Returns:
            Generator[Doc]:
                The output sequence of spacy documents with the extracted entities
        """
        instance_name = "ensure_serializable"
        try:
            self._nlp.get_pipe(instance_name)
        except KeyError:
            component_name = spacy.util.get_object_name(self._ensure_serializable)
            Language.component(name=component_name, func=self._ensure_serializable)  # type: ignore
            self._nlp.add_pipe(component_name, name=instance_name, last=True)

        n_process = n_process if n_process is not None else max(cpu_count() - 1, 1)
        batch_size = batch_size if batch_size is not None else 1000

        # If n_process < 0, multiprocessing will be either conducted inside pipeline components based on the con(when
        # 'parallel' is set to True) or not happen at all (when 'parallel' is set to False). Otherwise, multiprocessing
        # will be conducted at the pipeline level, i.e., texts will be processed sequentially by each pipeline component.
        if n_process < 0:
            inner_parallel = True
            n_process = 1
        else:
            inner_parallel = False

        component_cfg = {
            tag_skip_and_punct.name: {  # type: ignore
                'parallel': inner_parallel
            },
            TokenNormalizer.name: {
                'parallel': inner_parallel
            },
            NER.name: {
                'parallel': inner_parallel
            },
            Linker.name: {
                'parallel': inner_parallel
            }
        }

        return self._nlp.pipe(texts,    # type: ignore
                             n_process=n_process,
                             batch_size=batch_size,
                             component_cfg=component_cfg)

    def set_error_handler(self, error_handler: Callable) -> None:
        self._nlp.set_error_handler(error_handler)

    def reset_error_handler(self) -> None:
        self._nlp.set_error_handler(raise_error)

    def force_remove(self, component_name: str) -> None:
        try:
            self._nlp.remove_pipe(component_name)
        except ValueError:
            pass

    def destroy(self) -> None:
        del self._nlp
        gc.collect()

    @property
    def spacy_nlp(self) -> Language:
        """The spaCy Language object."""
        return self._nlp

    @staticmethod
    def _ensure_serializable(doc: Doc) -> Doc:
        return PipeRunner.serialize_entities(doc)

    def __call__(self, text: Union[str, Iterable[str]]) -> Union[Doc, List[Doc]]:
        if isinstance(text, str):
            return self._nlp(text) if len(text) > 0 else None  # type: ignore
        elif isinstance(text, Iterable):
            docs = []
            for t in text if isinstance(text, types.GeneratorType) else tqdm(text, total=len(list(text))):
                try:
                    doc = self._nlp(t) if isinstance(t, str) and len(t) > 0 else None
                except Exception as e:
                    logger.warning("Exception raised when processing text: %s", t[:50] + "..." if isinstance(t, str) else t)
                    logger.warning(e, exc_info=True, stack_info=True)
                    doc = None
                docs.append(doc)
            return docs  # type: ignore
        else:
            logger.error("The input text should be either a string or a sequence of strings but got: %s", type(text))
            return None
