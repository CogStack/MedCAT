from spacy.tokens import Span, Doc
from typing import Optional, List
from medcat.cdb import CDB
from enum import Enum, auto


class LabelStyle(Enum):
    short = auto()
    long = auto()


def map_ents_to_groups(cdb: CDB, doc: Doc) -> None:
    for ent in doc.ents:
        ent._.cui = cdb.addl_info['cui2group'].get(ent._.cui, ent._.cui)


def make_pretty_labels(cdb: CDB, doc: Doc, style: Optional[LabelStyle] = None) -> None:
    ents = list(doc.ents)

    n_ents = []
    for ent in ents:
        if style == LabelStyle.short:
            label = ent._.cui
        elif style == LabelStyle.long:
            label = "{} | {} | {:.2f}".format(ent._.cui, cdb.get_name(ent._.cui), ent._.context_similarity)
        else:
            label = 'concept'

        n_ent = Span(doc, ent.start, ent.end, label)
        for attr in ent._.__dict__['_extensions'].keys():
            setattr(n_ent._, attr, getattr(ent._, attr))
        n_ents.append(n_ent)

    doc.ents = n_ents  # type: ignore


def create_main_ann(cdb: CDB, doc: Doc, tuis: Optional[List] = None) -> None:
    # TODO: Separate into another piece of the pipeline
    """Creates annotation in the spacy ents list
    from all the annotations for this document.

    Args:
        cdb (CDB): The Context Databse.
        doc (Doc): Spacy document.
        tuis (Optional[List], optional): The type IDs. Defaults to None.
    """
    doc._.ents.sort(key=lambda x: len(x.text), reverse=True)

    tkns_in = set()
    main_anns = []
    for ent in doc._.ents:
        if tuis is None or ent._.tui in tuis:
            to_add = True
            for tkn in ent:
                if tkn in tkns_in:
                    to_add = False
            if to_add:
                for tkn in ent:
                    tkns_in.add(tkn)
                main_anns.append(ent)

    doc.ents = list(doc.ents) + main_anns  # type: ignore
