import logging
import joblib
from typing import Iterable, Generator
from spacy.tokens import Doc
from spacy.tokens.underscore import Underscore
from spacy.util import minibatch
from medcat.ner.vocab_based_annotator import maybe_annotate_name


class NER(object):
    r'''
    '''
    log = logging.getLogger(__name__)

    # Custom pipeline component name
    name = 'cat_ner'

    def __init__(self, cdb, config):
        self.config = config
        self.cdb = cdb

    def pipe(self, stream: Iterable[Doc], batch_size: int = None, n_process: int = None) -> Generator[Doc, None, None]:
        batch_size = self.config.ner['batch_size'] if batch_size is None else batch_size
        n_process = self.config.ner['n_process'] if n_process is None else n_process
        execute = joblib.Parallel(n_jobs=n_process)
        for docs in minibatch(stream, size=batch_size):
            try:
                run_pipe_on_one = joblib.delayed(NER._run_pipe_on_one)
                tasks = (run_pipe_on_one(self, doc, Underscore.get_state()) for doc in docs)
                yield from execute(tasks)
            except Exception as e:
                self.log.warning(e, stack_info=True)
                self.log.warning("Docs contained in the failed mini batch:")
                for doc in docs:
                    if hasattr(doc, "text"):
                        self.log.warning("{}...".format(doc.text[:50]))
                yield from docs

    def __call__(self, doc):
        r''' Detect candidates for concepts - linker will then be able to do the rest. It adds `entities` to the
        doc._.ents and each entity can have the entitiy._.link_candidates - that the linker will resolve.

        Args:
            doc (`spacy.tokens.Doc`):
                Spacy document to be annotated with named entities.
        Return
            doc (`spacy.tokens.Doc`):
                Spacy document with detected entities.
        '''
        # Just take the tokens we need
        _doc = [tkn for tkn in doc if not tkn._.to_skip]
        for i in range(len(_doc)):
            tkn = _doc[i]
            tkns = [tkn]
            #name_versions = [tkn.lower_, tkn._.norm]
            name_versions = [tkn._.norm, tkn.lower_]
            name = ""

            for name_version in name_versions:
                if name_version in self.cdb.snames:
                    if name:
                        name = name + self.config.general['separator'] + name_version
                    else:
                        name = name_version
                    break
            if name in self.cdb.name2cuis and not tkn.is_stop:
                maybe_annotate_name(name, tkns, doc, self.cdb, self.config)

            if name: # There has to be at least something appended to the name to go forward
                for j in range(i+1, len(_doc)):
                    if _doc[j].i - _doc[j-1].i - 1 > self.config.ner['max_skip_tokens']:
                         # Do not allow to skip more than limit
                         break
                    tkn = _doc[j]
                    tkns.append(tkn)
                    name_versions = [tkn._.norm, tkn.lower_]

                    name_changed = False
                    name_reverse = None
                    for name_version in name_versions:
                        _name = name + self.config.general['separator'] + name_version
                        if _name in self.cdb.snames:
                            # Append the name and break
                            name = _name
                            name_changed = True
                            break

                        if self.config.ner.get('try_reverse_word_order', False):
                            _name_reverse = name_version + self.config.general['separator'] + name
                            if _name_reverse in self.cdb.snames:
                                # Append the name and break
                                name_reverse = _name_reverse

                    if name_changed:
                        if name in self.cdb.name2cuis:
                            maybe_annotate_name(name, tkns, doc, self.cdb, self.config)
                    elif name_reverse is not None:
                        if name_reverse in self.cdb.name2cuis:
                            maybe_annotate_name(name_reverse, tkns, doc, self.cdb, self.config)
                    else:
                        break

        return doc

    @staticmethod
    def _run_pipe_on_one(ner, doc, underscore_state):
        Underscore.load_state(underscore_state)
        return ner(doc)
