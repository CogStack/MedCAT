from typing import Protocol, Tuple, List, Dict, Optional, Set, Iterator, Union, Callable, cast, Any

from copy import deepcopy

from medcat.utils.checkpoint import Checkpoint
from medcat.utils.cdb_state import captured_state_cdb

from medcat.stats.stats import get_stats
from medcat.stats.mctexport import MedCATTrainerExport, MedCATTrainerExportProject
from medcat.stats.mctexport import MedCATTrainerExportDocument, MedCATTrainerExportAnnotation
from medcat.stats.mctexport import count_all_annotations, count_all_docs
from medcat.stats.mctexport import iter_anns, iter_docs, MedCATTrainerExportProjectInfo



class CDBLike(Protocol):
    pass


class CATLike(Protocol):

    @property
    def cdb(self) -> CDBLike:
        pass

    def train_supervised_raw(self,
                             data: Dict[str, List[Dict[str, dict]]],
                             reset_cui_count: bool = False,
                             nepochs: int = 1,
                             print_stats: int = 0,
                             use_filters: bool = False,
                             terminate_last: bool = False,
                             use_overlaps: bool = False,
                             use_cui_doc_limit: bool = False,
                             test_size: float = 0,
                             devalue_others: bool = False,
                             use_groups: bool = False,
                             never_terminate: bool = False,
                             train_from_false_positives: bool = False,
                             extra_cui_filter: Optional[Set] = None,
                             retain_extra_cui_filter: bool = False,
                             checkpoint: Optional[Checkpoint] = None,
                             retain_filters: bool = False,
                             is_resumed: bool = False) -> Tuple:
        pass


class FoldCreator:
    """The FoldCreator based on a MCT export.

    Args:
        mct_export (MedCATTrainerExport): The MCT export dict.
        nr_of_folds (int): Number of folds to create.
        use_annotations (bool): Whether to fold on number of annotations or documents.
    """

    def __init__(self, mct_export: MedCATTrainerExport, nr_of_folds: int,
                 use_annotations: bool) -> None:
        self.mct_export = mct_export
        self.nr_of_folds = nr_of_folds
        self.use_annotations = use_annotations
        self._targets: Union[
            Iterator[Tuple[MedCATTrainerExportProjectInfo, MedCATTrainerExportDocument, MedCATTrainerExportAnnotation]],
            Iterator[Tuple[MedCATTrainerExportProjectInfo, MedCATTrainerExportDocument]]
        ]
        self._adder: Union[
            Callable[[MedCATTrainerExportProject, MedCATTrainerExportDocument, MedCATTrainerExportAnnotation], None],
            Callable[[MedCATTrainerExportProject, MedCATTrainerExportDocument], None]
        ]
        if self.use_annotations:
            self.total = count_all_annotations(self.mct_export)
            self._targets = iter_anns(self.mct_export)
            self._adder = self._add_target_ann
        else:
            self.total = count_all_docs(self.mct_export)
            self._targets = iter_docs(self.mct_export)
            self._adder = self._add_target_doc
        self.per_fold = self._init_per_fold()

    def _add_target_doc(self, project: MedCATTrainerExportProject,
                        doc: MedCATTrainerExportDocument) -> None:
        project['documents'].append(doc)

    def _find_or_add_doc(self, project: MedCATTrainerExportProject, orig_doc: MedCATTrainerExportDocument
                         ) -> MedCATTrainerExportDocument:
        for existing_doc in project['documents']:
            if existing_doc['name'] == orig_doc['name']:
                return existing_doc
        new_doc: MedCATTrainerExportDocument = deepcopy(orig_doc)
        new_doc['annotations'].clear()
        project['documents'].append(new_doc)
        return new_doc

    def _add_target_ann(self, project: MedCATTrainerExportProject, orig_doc: MedCATTrainerExportDocument,
                        ann: MedCATTrainerExportAnnotation) -> None:
        cur_doc: MedCATTrainerExportDocument = self._find_or_add_doc(project, orig_doc)
        cur_doc['annotations'].append(ann)

    def _init_per_fold(self) -> List[int]:
        per_fold = [self.total // self.nr_of_folds for _ in range(self.nr_of_folds)]
        total = sum(per_fold)
        if total < self.total:
            per_fold[-1] += self.total - total
        if any(pf <= 0 for pf in per_fold):
            raise ValueError(f"Failed to calculate per-fold items. Got: {per_fold}")
        return per_fold

    def _create_fold(self, fold_nr: int) -> MedCATTrainerExport:
        per_fold = self.per_fold[fold_nr]
        cur_fold: MedCATTrainerExport = {
            'projects': []
        }
        cur_project: Optional[MedCATTrainerExportProject] = None
        included = 0
        for target in self._targets:
            (proj_name, proj_id, proj_cuis, proj_tuis), *target_info = target
            if not cur_project or cur_project['name'] != proj_name:
                # first or new project
                cur_project = cast(MedCATTrainerExportProject, {
                    'name': proj_name,
                    'id': proj_id,
                    'cuis': proj_cuis,
                    'documents': [],
                })
                # NOTE: Some MCT exports don't declare TUIs
                if proj_tuis is not None:
                    cur_project['tuis'] = proj_tuis
                cur_fold['projects'].append(cur_project)
            self._adder(cur_project, *target_info)
            included += 1
            if included == per_fold:
                break
            if included > per_fold:
                raise ValueError("Got a larger fold than expected. "
                                 f"Expected {per_fold}, got {included}")
        return cur_fold


    def create_folds(self) -> List[MedCATTrainerExport]:
        return [
            self._create_fold(fold_nr) for fold_nr in range(self.nr_of_folds)
        ]


def get_per_fold_metrics(cat: CATLike, folds: List[MedCATTrainerExport],
                         *args, **kwargs) -> List[Tuple]:
    metrics = []
    for fold_nr, cur_fold in enumerate(folds):
        others = list(folds)
        others.pop(fold_nr)
        with captured_state_cdb(cat.cdb):
            for other in others:
                cat.train_supervised_raw(cast(Dict[str, Any], other), *args, **kwargs)
            stats = get_stats(cat, cast(Dict[str, Any], cur_fold), do_print=False)
            metrics.append(stats)
    return metrics


def _update_all_weighted_average(joined: List[Dict[str, Tuple[int, float]]],
                single: List[Dict[str, float]], cui2count: Dict[str, int]) -> None:
    if len(joined) != len(single):
        raise ValueError(f"Incompatible lists. Joined {len(joined)} and single {len(single)}")
    for j, s in zip(joined, single):
        _update_one_weighted_average(j, s, cui2count)


def _update_one_weighted_average(joined: Dict[str, Tuple[int, float]],
                one: Dict[str, float],
                cui2count: Dict[str, int]) -> None:
    for k in one:
        if k not in joined:
            joined[k] = (0, 0)
        prev_w, prev_val = joined[k]
        new_w, new_val = cui2count[k], one[k]
        total_w = prev_w + new_w
        total_val = (prev_w * prev_val + new_w * new_val) / total_w
        joined[k] = (total_w, total_val)


def _update_all_add(joined: List[Dict[str, int]], single: List[Dict[str, int]]) -> None:
    if len(joined) != len(single):
        raise ValueError(f"Incompatible number of stuff: {len(joined)} vs {len(single)}")
    for j, s in zip(joined, single):
        for k, v in s.items():
            j[k] = j.get(k, 0) + v


def _merge_examples(all_examples: Dict, cur_examples: Dict) -> None:
    for ex_type, ex_dict in cur_examples.items():
        if ex_type not in all_examples:
            all_examples[ex_type] = {}
        per_type_examples = all_examples[ex_type]
        for ex_cui, cui_examples_list in ex_dict.items():
            if ex_cui not in per_type_examples:
                per_type_examples[ex_cui] = []
            per_type_examples[ex_cui].extend(cui_examples_list)


def get_metrics_mean(metrics: List[Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]]
                     ) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
    """The the mean of the provided metrics.

    Args:
        metrics (List[Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]): The metrics.

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
    # additives
    all_fps: Dict[str, int] = {}
    all_fns: Dict[str, int] = {}
    all_tps: Dict[str, int] = {}
    # weighted-averages
    all_cui_prec: Dict[str, Tuple[int, float]] = {}
    all_cui_rec: Dict[str, Tuple[int, float]] = {}
    all_cui_f1: Dict[str, Tuple[int, float]] = {}
    # additive
    all_cui_counts: Dict[str, int] = {}
    # combined
    all_additives = [
        all_fps, all_fns, all_tps, all_cui_counts
    ]
    all_weighted_averages = [
        all_cui_prec, all_cui_rec, all_cui_f1
    ]
    # examples
    all_examples: dict = {}
    for current in metrics:
        cur_wa: list = list(current[3:-2])
        cur_counts = current[-2]
        _update_all_weighted_average(all_weighted_averages, cur_wa, cur_counts)
        # update ones that just need to be added up
        cur_adds = list(current[:3]) + [cur_counts]
        _update_all_add(all_additives, cur_adds)
        # merge examples
        cur_examples = current[-1]
        _merge_examples(all_examples, cur_examples)
    cui_prec: Dict[str, float] = {}
    cui_rec: Dict[str, float] = {}
    cui_f1: Dict[str, float] = {}
    final_wa = [
        cui_prec, cui_rec, cui_f1
    ]
    # just remove the weight / count
    for df, d in zip(final_wa, all_weighted_averages):
        for k, v in d.items():
            df[k] = v[1]  # only the value, ingore the weight
    return (all_fps, all_fns, all_tps, final_wa[0], final_wa[1], final_wa[2],
            all_cui_counts, all_examples)


def get_k_fold_stats(cat: CATLike, mct_export_data: MedCATTrainerExport, k: int = 3,
                     use_annotations: bool = False, *args, **kwargs) -> Tuple:
    """Get the k-fold stats for the model with the specified data.

    First this will split the MCT export into `k` folds. You can do
    this either per document or per-annotation.

    For each of the `k` folds, it will start from the base model,
    train it with with the other `k-1` folds and record the metrics.
    After that the base model state is restored before doing the next fold.
    After all the folds have been done, the metrics are averaged.

    Args:
        cat (CATLike): The model pack.
        mct_export_data (MedCATTrainerExport): The MCT export.
        k (int): The number of folds. Defaults to 3.
        use_annotations (bool): Whether to use annodations or docs. Defaults to False (docs).
        *args: Arguments passed to the `CAT.train_supervised_raw` method.
        **kwargs: Keyword arguments passed to the `CAT.train_supervised_raw` method.

    Returns:
        Tuple: The averaged metrics.
    """
    creator = FoldCreator(mct_export_data, k, use_annotations=use_annotations)
    folds = creator.create_folds()
    per_fold_metrics = get_per_fold_metrics(cat, folds, *args, **kwargs)
    return get_metrics_mean(per_fold_metrics)
