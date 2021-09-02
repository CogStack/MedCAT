import types

import spacy
import gc
import logging
from spacy.tokens import Token, Doc, Span
from spacy.tokenizer import Tokenizer
from spacy.language import Language
from spacy.util import raise_error
from tqdm.autonotebook import tqdm

from medcat.linking.context_based_linker import Linker
from medcat.meta_cat import MetaCAT
from medcat.ner.vocab_based_ner import NER
from medcat.utils.normalizers import TokenNormalizer, BasicSpellChecker
from medcat.utils.loggers import add_handlers
from medcat.config import Config


from typing import List, Optional, Union, Iterable, Callable, Generator
from multiprocessing import cpu_count


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
    log = logging.getLogger(__package__)
    # Add file and console handlers
    log = add_handlers(log)
    def __init__(self, tokenizer: Tokenizer, config: Config):
        self.nlp = spacy.load(config.general['spacy_model'], disable=config.general['spacy_disabled_components'])
        if config.preprocessing['stopwords'] is not None:
            self.nlp.Defaults.stop_words = set(config.preprocessing['stopwords'])
        self.nlp.tokenizer = tokenizer(self.nlp)
        self.config = config
        # Set log level
        self.log.setLevel(self.config.general['log_level'])

    def add_tagger(self, tagger: Callable, name: Optional[str] = None, additional_fields: List[str] = []) -> None:
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
        component_factory_name = spacy.util.get_object_name(tagger)
        name = name if name is not None else component_factory_name
        Language.factory(name=component_factory_name, default_config={"config": self.config}, func=tagger)
        self.nlp.add_pipe(component_factory_name, name=name, first=True)

        # Add custom fields needed for this usecase
        Token.set_extension('to_skip', default=False, force=True)

        # Add any additional fields that are required
        for field in additional_fields:
            Token.set_extension(field, default=False, force=True)

    def add_token_normalizer(self, config: Config, name: Optional[str] = None, spell_checker: Optional[BasicSpellChecker] = None) -> None:
        token_normalizer = TokenNormalizer(spell_checker=spell_checker, config=config)
        component_name = spacy.util.get_object_name(token_normalizer)
        name = name if name is not None else component_name
        Language.component(name=component_name, func=token_normalizer)
        self.nlp.add_pipe(component_name, name=name, last=True)

        # Add custom fields needed for this usecase
        Token.set_extension('norm', default=None, force=True)

    def add_ner(self, ner: NER, name: Optional[str] = None) -> None:
        r''' Add NER from CAT to the pipeline, will also add the necessary fields
        to the document and Span objects.

        '''
        component_name = spacy.util.get_object_name(ner)
        name = name if name is not None else component_name
        Language.component(name=component_name, func=ner)
        self.nlp.add_pipe(component_name, name=name, last=True)

        Doc.set_extension('ents', default=[], force=True)
        Span.set_extension('confidence', default=-1, force=True)
        Span.set_extension('id', default=0, force=True)

        # Do not set this property if a vocabulary apporach is not used, this name must
        #refer to a name2cuis in the cdb.
        Span.set_extension('detected_name', default=None, force=True)
        Span.set_extension('link_candidates', default=None, force=True)

    def add_linker(self, linker: Linker, name: Optional[str] = None) -> None:
        r''' Add entity linker to the pipeline, will also add the necessary fields
        to Span object.

        linker (object/function):
            Any object/function created based on the requirements for a spaCy pipeline components. Have
            a look at https://spacy.io/usage/processing-pipelines#custom-components
        '''
        component_name = spacy.util.get_object_name(linker)
        name = name if name is not None else component_name
        Language.component(name=component_name, func=linker)
        self.nlp.add_pipe(component_name, name=name, last=True)
        Span.set_extension('cui', default=-1, force=True)
        Span.set_extension('context_similarity', default=-1, force=True)

    def add_meta_cat(self, meta_cat: MetaCAT, name: Optional[str] = None) -> None:
        component_name = spacy.util.get_object_name(meta_cat)
        name = name if name is not None else component_name
        Language.component(name=component_name, func=meta_cat)
        self.nlp.add_pipe(component_name, name=name, last=True)

        # Only the meta_anns field is needed, it will be a dictionary 
        #of {category_name: value, ...}
        Span.set_extension('meta_anns', default=None, force=True)

    def batch_multi_process(self,
                            texts: Iterable[str],
                            n_process: Optional[int] = 1, # 1 because multiprocessing will be conducted inside pipeline components so as to work with multi-core GPUs.
                            batch_size: Optional[int] = None,
                            total: Optional[int] = None) -> Generator[Doc, None, None]:
        r''' Batch process a list of texts in parallel.

        Args:
            texts (`Iterable[str]`):
                The input sequence of texts to process.
            n_process (`int`):
                The number of processes running in parallel. Defaults to max(mp.cpu_count() - 1, 1).
            batch_size (`int`):
                The number of texts to buffer. Defaults to 1000.
            total (`int`):
                The number of texts in total.

        Return:
            Generator[Doc]:
                The output sequence of spacy documents with the extracted entities
        '''
        instance_name = "ensure_serializable"
        try:
            self.nlp.get_pipe(instance_name)
        except KeyError:
            component_name = spacy.util.get_object_name(self._ensure_serializable)
            Language.component(name=component_name, func=self._ensure_serializable)
            self.nlp.add_pipe(component_name, name=instance_name, last=True)

        n_process = n_process if n_process is not None else max(cpu_count() - 1, 1)
        batch_size = batch_size if batch_size is not None else 1000

        return self.nlp.pipe(texts if total is None else tqdm(texts, total=total), n_process=n_process, batch_size=batch_size)

    def set_error_handler(self, error_handler):
        self.nlp.set_error_handler(error_handler)

    def reset_error_handler(self):
        self.nlp.set_error_handler(raise_error)

    def force_remove(self, component_name: str) -> None:
        try:
            self.nlp.remove_pipe(component_name)
        except ValueError:
            pass

    def destroy(self):
        del self.nlp
        gc.collect()

    @staticmethod
    def _ensure_serializable(doc: Doc) -> Doc:
        new_ents = []
        for ent in doc._.ents:
            serializable = {
                "start": ent.start,
                "end": ent.end,
                "label": ent.label,
                "cui": ent._.cui,
                "detected_name": ent._.detected_name,
                "context_similarity": ent._.context_similarity,
                "id": ent._.id
            }
            if hasattr(ent._, 'meta_anns') and ent._.meta_anns:
                serializable['meta_anns'] = ent._.meta_anns
            new_ents.append(serializable)
        doc._.ents.clear()
        doc._.ents = new_ents
        return doc

    def __call__(self, text: Union[str, Iterable[str]]) -> Union[Doc, List[Doc]]:
        if isinstance(text, str):
            return self.nlp(text) if len(text) > 0 else None
        elif isinstance(text, Iterable):
            docs = []
            for t in text if isinstance(text, types.GeneratorType) else tqdm(text, total=len(text)):
                try:
                    doc = self.nlp(t) if isinstance(t, str) and len(t) > 0 else None
                except Exception as e:
                    self.log.warning("Exception raised when processing text: {}".format(t[:50] + "..." if isinstance(t, str) else t))
                    self.log.warning(e, stack_info=True)
                    doc = None
                docs.append(doc)
            return docs
        else:
            self.log.error("The input text should be either a string or a sequence of strings but got: {}".format(type(text)))
            return None
