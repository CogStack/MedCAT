import logging
from joblib import Parallel, delayed
from typing import Iterable, Generator, Tuple
from spacy.tokens import Doc
from spacy.tokens.underscore import Underscore
from spacy.util import minibatch


class PipeRunner(object):

    log = logging.getLogger(__name__)
    _execute = None
    _time_out_in_secs = 3600

    def __init__(self, workers: int, batch_size: int):
        self.batch_size = batch_size
        PipeRunner._execute = Parallel(n_jobs=workers, timeout=PipeRunner._time_out_in_secs)

    def pipe(self, stream: Iterable[Doc], **kwargs) -> Generator[Doc, None, None]:
        if kwargs.get("n_process") == 1:
            # Multiprocessing will be conducted inside pipeline components so as to work with multi-core GPUs.
            for docs in minibatch(stream, size=self.batch_size):
                try:
                    run_pipe_on_one = delayed(self._run_pipe_on_one)
                    tasks = (run_pipe_on_one(self, doc, Underscore.get_state()) for doc in docs)
                    yield from PipeRunner._execute(tasks)
                except Exception as e:
                    self.log.warning(e, stack_info=True)
                    self.log.warning("Docs contained in the failed mini batch:")
                    for doc in docs:
                        if hasattr(doc, "text"):
                            self.log.warning("{}...".format(doc.text[:50]))
                    yield from docs
        else:
            # Multiprocessing will be conducted at the pipeline level.
            # And texts will be processed sequentially inside components.
            for doc in stream:
                yield self(doc)

    def __call__(self, doc: Doc) -> Doc:
        raise NotImplementedError("__call__ is not implemented.")

    @staticmethod
    def _run_pipe_on_one(pipe_runner: "PipeRunner", doc: Doc, underscore_state: Tuple) -> "PipeRunner":
        Underscore.load_state(underscore_state)
        return pipe_runner(doc)
