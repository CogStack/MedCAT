import logging
import gc
from joblib import Parallel, delayed
from typing import Iterable, Generator, Tuple, Callable
from spacy.tokens import Doc, Span
from spacy.tokens.underscore import Underscore
from spacy.pipeline import Pipe
from spacy.util import minibatch


class PipeRunner(Pipe):

    log = logging.getLogger(__name__)
    _execute = None
    _delayed = None
    _time_out_in_secs = 3600

    def __init__(self, workers: int):
        self.workers = workers

    def __call__(self, doc: Doc):
        raise NotImplementedError("Method __call__ has not been implemented.")

    def pipe(self, stream: Iterable[Doc], batch_size: int, **kwargs) -> Generator[Doc, None, None]:
        error_handler = self.get_error_handler()
        if kwargs.get("parallel", False):
            self._lazy_init_pool()
            for docs in minibatch(stream, size=self.workers):
                docs = [PipeRunner.serialize_entities(doc) for doc in docs]
                try:
                    tasks = (PipeRunner._delayed(self.__call__, doc, Underscore.get_state()) for doc in docs)
                    for output_doc in PipeRunner._execute(tasks):
                        yield PipeRunner.deserialize_entities(output_doc)
                except Exception as e:
                    error_handler(self.name, self, docs, e)
                    yield from [None] * len(docs)
        else:
            for doc in stream:
                try:
                    yield self(doc)
                except Exception as e:
                    error_handler(self.name, self, [doc], e)
                    yield None

    @staticmethod
    def serialize_entities(doc: Doc):
        new_ents = []
        for ent in doc._.ents:
            serializable = {
                "start": ent.start,
                "end": ent.end,
                "label": ent.label_,
                "cui": ent._.cui,
                "detected_name": ent._.detected_name,
                "context_similarity": ent._.context_similarity,
                "link_candidates": ent._.link_candidates,
                "confidence": ent._.confidence,
                "id": ent._.id
            }
            if hasattr(ent._, 'meta_anns') and ent._.meta_anns:
                serializable['meta_anns'] = ent._.meta_anns
            new_ents.append(serializable)
        doc._.ents.clear()
        gc.collect()
        doc._.ents = new_ents
        return doc

    @staticmethod
    def deserialize_entities(doc: Doc):
        new_ents = []
        for ent in doc._.ents:
            ent_span = Span(doc, ent['start'], ent['end'], label=ent['label'])
            ent_span._.cui = ent['cui']
            ent_span._.detected_name = ent['detected_name']
            ent_span._.context_similarity = ent['context_similarity']
            ent_span._.link_candidates = ent['link_candidates']
            ent_span._.confidence = ent['confidence']
            ent_span._.id = ent['id']
            if 'meta_anns' in ent:
                ent_span._.meta_anns = ent['meta_anns']
            new_ents.append(ent_span)
        doc._.ents.clear()
        gc.collect()
        doc._.ents = new_ents
        return doc

    @staticmethod
    def _run_pipe_on_one(call: Callable, doc: Doc, underscore_state: Tuple) -> Doc:
        Underscore.load_state(underscore_state)
        doc = PipeRunner.deserialize_entities(doc)
        doc = call(doc)
        doc = PipeRunner.serialize_entities(doc)
        return doc

    def _lazy_init_pool(self):
        if PipeRunner._execute is None or self.workers > PipeRunner._execute.n_jobs:
            PipeRunner._execute = Parallel(n_jobs=self.workers, timeout=PipeRunner._time_out_in_secs)
        if PipeRunner._delayed is None:
            PipeRunner._delayed = delayed(PipeRunner._run_pipe_on_one)
