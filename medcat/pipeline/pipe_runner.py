import logging
from joblib import Parallel, delayed
from typing import Iterable, Generator, Tuple
from spacy.tokens import Doc
from spacy.tokens.underscore import Underscore
from spacy.pipeline import Pipe
from spacy.util import minibatch


class PipeRunner(Pipe):

    log = logging.getLogger(__name__)
    _execute = None
    _time_out_in_secs = 3600

    def __init__(self, workers: int):
        PipeRunner._execute = Parallel(n_jobs=workers, timeout=PipeRunner._time_out_in_secs)

    def pipe(self, stream: Iterable[Doc], batch_size: int, **kwargs) -> Generator[Doc, None, None]:
        error_handler = self.get_error_handler()
        if kwargs.get("parallel", False):
            for docs in minibatch(stream, size=batch_size):
                try:
                    run_pipe_on_one = delayed(self._run_pipe_on_one)
                    tasks = (run_pipe_on_one(self, doc, Underscore.get_state()) for doc in docs)
                    yield from PipeRunner._execute(tasks)
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
    def _run_pipe_on_one(pipe_runner: "PipeRunner", doc: Doc, underscore_state: Tuple) -> Doc:
        Underscore.load_state(underscore_state)
        return pipe_runner(doc)
