import logging
from spacy.tokens import Doc
from medcat.ner.vocab_based_annotator import maybe_annotate_name
from medcat.pipeline.pipe_runner import PipeRunner
from medcat.cdb import CDB
from medcat.config import Config


logger = logging.getLogger(__name__)


class NER(PipeRunner):

    # Custom pipeline component name
    name = 'cat_ner'

    # Override
    def __init__(self, cdb: CDB, config: Config) -> None:
        self.config = config
        self.cdb = cdb
        super().__init__(self.config.general.workers)

    # Override
    def __call__(self, doc: Doc) -> Doc:
        """Detect candidates for concepts - linker will then be able to do the rest. It adds `entities` to the
        doc._.ents and each entity can have the entity._.link_candidates - that the linker will resolve.

        Args:
            doc (Doc):
                Spacy document to be annotated with named entities.

        Returns:
            doc (Doc):
                Spacy document with detected entities.
        """
        # Just take the tokens we need
        _doc = [tkn for tkn in doc if not tkn._.to_skip]
        for i in range(len(_doc)):
            tkn = _doc[i]
            tkns = [tkn]
            #name_versions = [tkn.lower_, tkn._.norm]
            name_versions = [tkn._.norm, tkn.lower_]
            name = ""

            nv_in_snames = []
            nv_in_names = []
            for name_version in name_versions:
                # NOTE: if the entire token is an actual concept, we want to capture that
                #       previous implementation could fail in those cases
                if name_version in self.cdb.snames:
                    nv_in_snames.append(name_version)
                if name_version in self.cdb.name2cuis:
                    nv_in_names.append(name_version)
            if nv_in_names:
                # TODO: should we prefer 0th (i.e the normalised version) or last (the lower case version)
                name = nv_in_names[0]
            elif nv_in_snames:
                # TODO: should we prefer 0th (i.e the normalised version) or last (the lower case version)
                name = nv_in_snames[0]
            if name in self.cdb.name2cuis and not tkn.is_stop:
                maybe_annotate_name(name, tkns, doc, self.cdb, self.config)

            if name: # There has to be at least something appended to the name to go forward
                for j in range(i+1, len(_doc)):
                    if _doc[j].i - _doc[j-1].i - 1 > self.config.ner.max_skip_tokens:
                        # Do not allow to skip more than limit
                        break
                    tkn = _doc[j]
                    tkns.append(tkn)
                    name_versions = [tkn._.norm, tkn.lower_]

                    name_changed = False
                    name_reverse = None
                    for name_version in name_versions:
                        _name = name + self.config.general.separator + name_version
                        if _name in self.cdb.snames:
                            # Append the name and break
                            name = _name
                            name_changed = True
                            break

                        if self.config.ner.get('try_reverse_word_order', False):
                            _name_reverse = name_version + self.config.general.separator + name
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
