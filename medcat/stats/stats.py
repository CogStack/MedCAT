from typing import Dict, Optional, Set, Tuple, Callable, List, cast

from tqdm import tqdm
import traceback

from spacy.tokens import Doc

from medcat.utils.filters import set_project_filters
from medcat.utils.matutils import intersect_nonempty_set
from medcat.config import LinkingFilters


class StatsBuilder:

    def __init__(self,
                 filters: LinkingFilters,
                 addl_info: dict,
                 doc_getter: Callable[[Optional[str], bool], Optional[Doc]],
                 doc_annotation_getter: Callable[[dict], list],
                 cui2group: Dict[str, str],
                 cui2preferred_name: Dict[str, str],
                 cui2names: Dict[str, Set[str]],
                 use_project_filters: bool = False,
                 use_overlaps: bool = False,
                 use_cui_doc_limit: bool = False,
                 use_groups: bool = False,
                 extra_cui_filter: Optional[Set] = None) -> None:
        self.filters = filters
        self.addl_info = addl_info
        self.doc_getter = doc_getter
        self._get_doc_annotations = doc_annotation_getter
        self.cui2group = cui2group
        self.cui2preferred_name = cui2preferred_name
        self.cui2names = cui2names
        self.use_project_filters = use_project_filters
        self.use_overlaps = use_overlaps
        self.use_cui_doc_limit = use_cui_doc_limit
        self.use_groups = use_groups
        self.extra_cui_filter = extra_cui_filter
        self._reset_stats()

    def _reset_stats(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.fps: Dict = {}
        self.fns: Dict = {}
        self.tps: Dict = {}
        self.cui_prec: Dict = {}
        self.cui_rec: Dict = {}
        self.cui_f1: Dict = {}
        self.cui_counts: Dict = {}
        self.examples: Dict = {'fp': {}, 'fn': {}, 'tp': {}}
        self.fp_docs: Set = set()
        self.fn_docs: Set = set()

    def process_project(self, project: dict) -> None:
        self.filters.cuis = set()

        # Add extra filter if set
        set_project_filters(self.addl_info, self.filters, project, self.extra_cui_filter, self.use_project_filters)

        project_name = cast(str, project.get('name'))
        project_id = cast(str, project.get('id'))

        documents = project["documents"]
        for dind, doc in tqdm(
            enumerate(documents),
            desc="Stats document",
            total=len(documents),
            leave=False,
        ):
            self.process_document(project_name, project_id, doc)

    def process_document(self, project_name: str, project_id: str, doc: dict) -> None:
        anns = self._get_doc_annotations(doc)

        # Apply document level filtering, in this case project_filter is ignored while the extra_cui_filter is respected still
        if self.use_cui_doc_limit:
            _cuis = set([ann['cui'] for ann in anns])
            if _cuis:
                self.filters.cuis = intersect_nonempty_set(_cuis, self.extra_cui_filter)
            else:
                self.filters.cuis = {'empty'}

        spacy_doc: Doc = self.doc_getter(doc['text'])  # type: ignore

        if self.use_overlaps:
            p_anns = spacy_doc._.ents
        else:
            p_anns = spacy_doc.ents

        (anns_norm, anns_norm_neg,
         anns_examples, _) = self._preprocess_annotations(project_name, project_id, doc, anns)

        p_anns_norm, p_anns_examples = self._process_p_anns(project_name, project_id,
                                                            doc, p_anns)
        self._count_p_anns_norm(doc, anns_norm, anns_norm_neg,
                                p_anns_norm, p_anns_examples)
        self._process_anns_norm(doc, anns_norm, p_anns_norm, anns_examples)

    def _process_anns_norm(self, doc: dict, anns_norm: list, p_anns_norm: list,
                           anns_examples: list) -> None:
        for iann, ann in enumerate(anns_norm):
            if ann not in p_anns_norm:
                cui = ann[1]
                self.fn += 1
                self.fn_docs.add(doc.get('name', 'unk'))

                self.fns[cui] = self.fns.get(cui, 0) + 1
                self.examples['fn'][cui] = self.examples['fn'].get(cui, []) + [anns_examples[iann]]

    def _process_p_anns(self, project_name: str, project_id: str, doc: dict, p_anns: list) -> Tuple[list, list]:
        p_anns_norm = []
        p_anns_examples = []
        for ann in p_anns:
            cui = ann._.cui
            if self.use_groups:
                cui = self.cui2group.get(cui, cui)

            p_anns_norm.append((ann.start_char, cui))
            p_anns_examples.append(self._create_annoation_2(project_name, project_id, cui, doc, ann))
        return p_anns_norm, p_anns_examples

    def _count_p_anns_norm(self, doc: dict, anns_norm: list, anns_norm_neg: list,
                           p_anns_norm: list, p_anns_examples: list) -> None:
        for iann, ann in enumerate(p_anns_norm):
            cui = ann[1]
            if ann in anns_norm:
                self.tp += 1
                self.tps[cui] = self.tps.get(cui, 0) + 1

                example = p_anns_examples[iann]
                self.examples['tp'][cui] = self.examples['tp'].get(cui, []) + [example]
            else:
                self.fp += 1
                self.fps[cui] = self.fps.get(cui, 0) + 1
                self.fp_docs.add(doc.get('name', 'unk'))

                # Add example for this FP prediction
                example = p_anns_examples[iann]
                if ann in anns_norm_neg:
                    # Means that it really was annotated as negative
                    example['real_fp'] = True

                self.examples['fp'][cui] = self.examples['fp'].get(cui, []) + [example]

    def _create_annoation(self, project_name: str, project_id: str, cui: str, doc: dict, ann: Dict) -> Dict:
        return {"text": doc['text'][max(0, ann['start']-60):ann['end']+60],
                "cui": cui,
                "start": ann['start'],
                "end": ann['end'],
                "source value": ann['value'],
                "acc": 1,
                "project name": project_name,
                "document name": doc.get('name'),
                "project id": project_id,
                "document id": doc.get('id')}

    def _create_annoation_2(self, project_name: str, project_id: str, cui: str, doc: dict, ann) -> Dict:
        return {"text": doc['text'][max(0, ann.start_char-60):ann.end_char+60],
                "cui": cui,
                "start": ann.start_char,
                "end": ann.end_char,
                "source value": ann.text,
                "acc": float(ann._.context_similarity),
                "project name": project_name,
                "document name": doc.get('name'),
                "project id": project_id,
                "document id": doc.get('id')}

    def _preprocess_annotations(self, project_name: str, project_id: str,
                                doc: dict, anns: List[Dict]) -> Tuple[list, list, list, list]:
        anns_norm = []
        anns_norm_neg = []
        anns_examples = []
        anns_norm_cui = []
        for ann in anns:
            cui = ann['cui']
            if self.filters.check_filters(cui):
                if self.use_groups:
                    cui = self.cui2group.get(cui, cui)

                if ann.get('validated', True) and (not ann.get('killed', False) and not ann.get('deleted', False)):
                    anns_norm.append((ann['start'], cui))
                    anns_examples.append(self._create_annoation(project_name, project_id, cui, doc, ann))
                elif ann.get('validated', True) and (ann.get('killed', False) or ann.get('deleted', False)):
                    anns_norm_neg.append((ann['start'], cui))

                if ann.get("validated", True):
                    # This is used to test was someone annotating for this CUI in this document
                    anns_norm_cui.append(cui)
                    self.cui_counts[cui] = self.cui_counts.get(cui, 0) + 1
        return anns_norm, anns_norm_neg, anns_examples, anns_norm_cui

    def finalise_report(self, epoch: int, do_print: bool = True):
        try:
            prec = self.tp / (self.tp + self.fp)
            rec = self.tp / (self.tp + self.fn)
            f1 = 2*(prec*rec) / (prec + rec)
            if do_print:
                print("Epoch: {}, Prec: {}, Rec: {}, F1: {}\n".format(epoch, prec, rec, f1))
                print("Docs with false positives: {}\n".format("; ".join([str(x) for x in list(self.fp_docs)[0:10]])))
                print("Docs with false negatives: {}\n".format("; ".join([str(x) for x in list(self.fn_docs)[0:10]])))

            # Sort fns & prec
            fps = {k: v for k, v in sorted(self.fps.items(), key=lambda item: item[1], reverse=True)}
            fns = {k: v for k, v in sorted(self.fns.items(), key=lambda item: item[1], reverse=True)}
            tps = {k: v for k, v in sorted(self.tps.items(), key=lambda item: item[1], reverse=True)}


            # F1 per concept
            for cui in tps.keys():
                prec = tps[cui] / (tps.get(cui, 0) + fps.get(cui, 0))
                rec = tps[cui] / (tps.get(cui, 0) + fns.get(cui, 0))
                f1 = 2*(prec*rec) / (prec + rec)
                self.cui_prec[cui] = prec
                self.cui_rec[cui] = rec
                self.cui_f1[cui] = f1


            # Get top 10
            pr_fps = [(self.cui2preferred_name.get(cui,
                list(self.cui2names.get(cui, [cui]))[0]), cui, fps[cui]) for cui in list(fps.keys())[0:10]]
            pr_fns = [(self.cui2preferred_name.get(cui,
                list(self.cui2names.get(cui, [cui]))[0]), cui, fns[cui]) for cui in list(fns.keys())[0:10]]
            pr_tps = [(self.cui2preferred_name.get(cui,
                list(self.cui2names.get(cui, [cui]))[0]), cui, tps[cui]) for cui in list(tps.keys())[0:10]]

            if do_print:
                print("\n\nFalse Positives\n")
                for one in pr_fps:
                    print("{:70} - {:20} - {:10}".format(str(one[0])[0:69], str(one[1])[0:19], one[2]))
                print("\n\nFalse Negatives\n")
                for one in pr_fns:
                    print("{:70} - {:20} - {:10}".format(str(one[0])[0:69], str(one[1])[0:19], one[2]))
                print("\n\nTrue Positives\n")
                for one in pr_tps:
                    print("{:70} - {:20} - {:10}".format(str(one[0])[0:69], str(one[1])[0:19], one[2]))
                print("*"*110 + "\n")

        except Exception:
            traceback.print_exc()

    def unwrap(self) -> Tuple:
        return (self.fps, self.fns, self.tps,
                self.cui_prec, self.cui_rec, self.cui_f1,
                self.cui_counts, self.examples)

    @classmethod
    def from_cat(cls, cat,
                 local_filters: LinkingFilters,
                 use_project_filters: bool = False,
                 use_overlaps: bool = False,
                 use_cui_doc_limit: bool = False,
                 use_groups: bool = False,
                 extra_cui_filter: Optional[Set] = None) -> 'StatsBuilder':
        return StatsBuilder(filters=local_filters,
                            addl_info=cat.cdb.addl_info,
                            doc_getter=cat.__call__,
                            doc_annotation_getter=cat._get_doc_annotations,
                            cui2group=cat.cdb.addl_info['cui2group'],
                            cui2preferred_name=cat.cdb.cui2preferred_name,
                            cui2names=cat.cdb.cui2names,
                            use_project_filters=use_project_filters,
                            use_overlaps=use_overlaps,
                            use_cui_doc_limit=use_cui_doc_limit,
                            use_groups=use_groups,
                            extra_cui_filter=extra_cui_filter)


def get_stats(cat,
              data: Dict,
              epoch: int = 0,
              use_project_filters: bool = False,
              use_overlaps: bool = False,
              use_cui_doc_limit: bool = False,
              use_groups: bool = False,
              extra_cui_filter: Optional[Set] = None,
              do_print: bool = True) -> Tuple:
    """TODO: Refactor and make nice
    Print metrics on a dataset (F1, P, R), it will also print the concepts that have the most FP,FN,TP.

    Args:
        cat: (CAT):
            The model pack.
        data (Dict):
            The json object that we get from MedCATtrainer on export.
        epoch (int):
            Used during training, so we know what epoch is it.
        use_project_filters (bool):
            Each project in MedCATtrainer can have filters, do we want to respect those filters
            when calculating metrics.
        use_overlaps (bool):
            Allow overlapping entities, nearly always False as it is very difficult to annotate overlapping entities.
        use_cui_doc_limit (bool):
            If True the metrics for a CUI will be only calculated if that CUI appears in a document, in other words
            if the document was annotated for that CUI. Useful in very specific situations when during the annotation
            process the set of CUIs changed.
        use_groups (bool):
            If True concepts that have groups will be combined and stats will be reported on groups.
        extra_cui_filter(Optional[Set]):
            This filter will be intersected with all other filters, or if all others are not set then only this one will be used.
        do_print (bool):
            Whether to print stats out. Defaults to True.

    Returns:
        fps (dict):
            False positives for each CUI.
        fns (dict):
            False negatives for each CUI.
        tps (dict):
            True positives for each CUI.
        cui_prec (dict):
            Precision for each CUI.
        cui_rec (dict):
            Recall for each CUI.
        cui_f1 (dict):
            F1 for each CUI.
        cui_counts (dict):
            Number of occurrence for each CUI.
        examples (dict):
            Examples for each of the fp, fn, tp. Format will be examples['fp']['cui'][<list_of_examples>].
    """
    orig_filters = cat.config.linking.filters.copy_of()
    local_filters = cat.config.linking.filters
    builder = StatsBuilder.from_cat(cat,
                                    local_filters=local_filters,
                                    use_project_filters=use_project_filters,
                                    use_overlaps=use_overlaps,
                                    use_cui_doc_limit=use_cui_doc_limit,
                                    use_groups=use_groups,
                                    extra_cui_filter=extra_cui_filter)
    for pind, project in tqdm(enumerate(data['projects']), desc="Stats project", total=len(data['projects']), leave=False):
        builder.process_project(project)

    # this is the part that prints out the stats
    builder.finalise_report(epoch, do_print=do_print)

    cat.config.linking.filters = orig_filters

    return builder.unwrap()
