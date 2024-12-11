from typing import Protocol, Tuple, List, Dict, Optional, Set, Iterable, Callable, cast, Any, Union

from abc import ABC, abstractmethod
from enum import Enum, auto
from copy import deepcopy
from pydantic import BaseModel
from itertools import islice

import numpy as np

from medcat.utils.checkpoint import Checkpoint
from medcat.utils.cdb_state import captured_state_cdb

from medcat.stats.stats import get_stats
from medcat.stats.mctexport import MedCATTrainerExport, MedCATTrainerExportProject
from medcat.stats.mctexport import MedCATTrainerExportDocument, MedCATTrainerExportAnnotation
from medcat.stats.mctexport import count_all_annotations, count_all_docs, get_nr_of_annotations
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


class SplitType(Enum):
    """The split type."""
    DOCUMENTS = auto()
    """Split over number of documents."""
    ANNOTATIONS = auto()
    """Split over number of annotations."""
    DOCUMENTS_WEIGHTED = auto()
    """Split over number of documents based on the number of annotations.
    So essentially this ensures that the same document isn't in 2 folds
    while trying to more equally distribute documents with different number
    of annotations.
    For example:
        If we have 6 documents that we want to split into 3 folds.
        The number of annotations per document are as follows:
           [40, 40, 20, 10, 5, 5]
        If we were to split this trivially over documents, we'd end up
        with the 3 folds with number of annotations that are far from even:
           [80, 30, 10]
        However, if we use the annotations as weights, we would be able to
        create folds that have more evenly distributed annotations, e.g:
           [[D1,], [D2], [D3, D4, D5, D6]]
        where D# denotes the number of the documents, with the number of
        annotations being equal:
           [ 40, 40, 20 + 10 + 5 + 5 = 40]
    """


class FoldCreator(ABC):
    """The FoldCreator based on a MCT export.

    Args:
        mct_export (MedCATTrainerExport): The MCT export dict.
        nr_of_folds (int): Number of folds to create.
        use_annotations (bool): Whether to fold on number of annotations or documents.
    """

    def __init__(self, mct_export: MedCATTrainerExport, nr_of_folds: int) -> None:
        self.mct_export = mct_export
        self.nr_of_folds = nr_of_folds

    def _find_or_add_doc(self, project: MedCATTrainerExportProject, orig_doc: MedCATTrainerExportDocument
                         ) -> MedCATTrainerExportDocument:
        for existing_doc in project['documents']:
            if existing_doc['name'] == orig_doc['name']:
                return existing_doc
        new_doc: MedCATTrainerExportDocument = deepcopy(orig_doc)
        new_doc['annotations'].clear()
        project['documents'].append(new_doc)
        return new_doc

    def _create_new_project(self, proj_info: MedCATTrainerExportProjectInfo) -> MedCATTrainerExportProject:
        (proj_name, proj_id, proj_cuis, proj_tuis) = proj_info
        cur_project = cast(MedCATTrainerExportProject, {
            'name': proj_name,
            'id': proj_id,
            'cuis': proj_cuis,
            'documents': [],
            })
        # NOTE: Some MCT exports don't declare TUIs
        if proj_tuis is not None:
            cur_project['tuis'] = proj_tuis
        return cur_project

    def _create_export_with_documents(self, relevant_docs: Iterable[Tuple[MedCATTrainerExportProjectInfo,
                                                                          MedCATTrainerExportDocument]]) -> MedCATTrainerExport:
        export: MedCATTrainerExport = {
            "projects": []
        }
        # helper for finding projects per name
        used_projects: Dict[str, MedCATTrainerExportProject] = {}
        for proj_info, doc in relevant_docs:
            proj_name = proj_info[0]
            if proj_name not in used_projects:
                cur_project = self._create_new_project(proj_info) # TODO - make sure it's available
                export['projects'].append(cur_project)
                used_projects[proj_name] = cur_project
            else:
                cur_project = used_projects[proj_name]
            cur_project['documents'].append(doc)
        return export


    @abstractmethod
    def create_folds(self) -> List[MedCATTrainerExport]:
        """Create folds.

        Raises:
            ValueError: If something went wrong.

        Returns:
            List[MedCATTrainerExport]: The created folds.
        """


class SimpleFoldCreator(FoldCreator):

    def __init__(self, mct_export: MedCATTrainerExport, nr_of_folds: int,
                 counter: Callable[[MedCATTrainerExport], int]) -> None:
        super().__init__(mct_export, nr_of_folds)
        self._counter = counter
        self.total = self._counter(mct_export)
        self.per_fold = self._init_per_fold()

    def _init_per_fold(self) -> List[int]:
        per_fold = [self.total // self.nr_of_folds for _ in range(self.nr_of_folds)]
        total = sum(per_fold)
        if total < self.total:
            per_fold[-1] += self.total - total
        if any(pf <= 0 for pf in per_fold):
            raise ValueError(f"Failed to calculate per-fold items. Got: {per_fold}")
        return per_fold

    @abstractmethod
    def _create_fold(self, fold_nr: int) -> MedCATTrainerExport:
        pass

    def create_folds(self) -> List[MedCATTrainerExport]:
        return [
            self._create_fold(fold_nr) for fold_nr in range(self.nr_of_folds)
        ]



class PerDocsFoldCreator(FoldCreator):

    def __init__(self, mct_export: MedCATTrainerExport, nr_of_folds: int) -> None:
        super().__init__(mct_export, nr_of_folds)
        self.nr_of_docs = count_all_docs(self.mct_export)
        self.per_doc_simple = self.nr_of_docs // self.nr_of_folds
        self._all_docs = list(iter_docs(self.mct_export))

    def _create_fold(self, fold_nr: int) -> MedCATTrainerExport:
        start_nr = self.per_doc_simple * fold_nr
        # until the end for last fold, otherwise just the next set of docs
        end_nr = self.nr_of_docs if fold_nr == self.nr_of_folds - 1 else start_nr + self.per_doc_simple
        relevant_docs = self._all_docs[start_nr: end_nr]
        return self._create_export_with_documents(relevant_docs)

    def create_folds(self) -> List[MedCATTrainerExport]:
        return [
            self._create_fold(fold_nr) for fold_nr in range(self.nr_of_folds)
        ]


class PerAnnsFoldCreator(SimpleFoldCreator):

    def __init__(self, mct_export: MedCATTrainerExport, nr_of_folds: int) -> None:
        super().__init__(mct_export, nr_of_folds, count_all_annotations)

    def _add_target_ann(self, project: MedCATTrainerExportProject,
                        orig_doc: MedCATTrainerExportDocument,
                        ann: MedCATTrainerExportAnnotation) -> None:
        cur_doc: MedCATTrainerExportDocument = self._find_or_add_doc(project, orig_doc)
        cur_doc['annotations'].append(ann)

    def _targets(self, start_at: int) -> Iterable[Tuple[MedCATTrainerExportProjectInfo,
                                                        MedCATTrainerExportDocument,
                                                        MedCATTrainerExportAnnotation]]:
        return islice(iter_anns(self.mct_export), start_at, None)

    def _create_fold(self, fold_nr: int) -> MedCATTrainerExport:
        per_fold = self.per_fold[fold_nr]
        already_used = sum(self.per_fold[fn] for fn in range(fold_nr))
        cur_fold: MedCATTrainerExport = {
            'projects': []
        }
        cur_project: Optional[MedCATTrainerExportProject] = None
        included = 0
        for target in self._targets(already_used):
            proj_info, cur_doc, cur_ann = target
            proj_name = proj_info[0]
            if not cur_project or cur_project['name'] != proj_name:
                # first or new project
                cur_project = self._create_new_project(proj_info)
                cur_fold['projects'].append(cur_project)
            self._add_target_ann(cur_project, cur_doc, cur_ann)
            included += 1
            if included == per_fold:
                break
            if included > per_fold:
                raise ValueError("Got a larger fold than expected. "
                                 f"Expected {per_fold}, got {included}")
        return cur_fold


class WeightedDocumentsCreator(FoldCreator):

    def __init__(self, mct_export: MedCATTrainerExport, nr_of_folds: int,
                 weight_calculator: Callable[[MedCATTrainerExportDocument], int]) -> None:
        super().__init__(mct_export, nr_of_folds)
        self._weight_calculator = weight_calculator
        docs = [(doc, self._weight_calculator(doc[1])) for doc in iter_docs(self.mct_export)]
        # descending order in weight
        self._weighted_docs = sorted(docs, key=lambda d: d[1], reverse=True)

    def create_folds(self) -> List[MedCATTrainerExport]:
        doc_folds: List[List[Tuple[MedCATTrainerExportProjectInfo, MedCATTrainerExportDocument]]]
        doc_folds = [[] for _ in range(self.nr_of_folds)]
        fold_weights = [0] * self.nr_of_folds

        for item, weight in self._weighted_docs:
            # Find the subset with the minimum total weight
            min_subset_idx = np.argmin(fold_weights)
            # add the most heavily weighted document
            doc_folds[min_subset_idx].append(item)
            fold_weights[min_subset_idx] += weight

        return [self._create_export_with_documents(docs) for docs in doc_folds]


def get_fold_creator(mct_export: MedCATTrainerExport,
                     nr_of_folds: int,
                     split_type: SplitType) -> FoldCreator:
    """Get the appropriate fold creator.

    Args:
        mct_export (MedCATTrainerExport): The MCT export.
        nr_of_folds (int): Number of folds to use.
        split_type (SplitType): The type of split to use.

    Raises:
        ValueError: In case of an unknown split type.

    Returns:
        FoldCreator: The corresponding fold creator.
    """
    if split_type is SplitType.DOCUMENTS:
        return PerDocsFoldCreator(mct_export=mct_export, nr_of_folds=nr_of_folds)
    elif split_type is SplitType.ANNOTATIONS:
        return PerAnnsFoldCreator(mct_export=mct_export, nr_of_folds=nr_of_folds)
    elif split_type is SplitType.DOCUMENTS_WEIGHTED:
        return WeightedDocumentsCreator(mct_export=mct_export, nr_of_folds=nr_of_folds,
                                        weight_calculator=get_nr_of_annotations)
    else:
        raise ValueError(f"Unknown Split Type: {split_type}")


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


def _merge_examples(all_examples: Dict, cur_examples: Dict) -> None:
    for ex_type, ex_dict in cur_examples.items():
        if ex_type not in all_examples:
            all_examples[ex_type] = {}
        per_type_examples = all_examples[ex_type]
        for ex_cui, cui_examples_list in ex_dict.items():
            if ex_cui not in per_type_examples:
                per_type_examples[ex_cui] = []
            per_type_examples[ex_cui].extend(cui_examples_list)


# helper types
IntValuedMetric = Union[
    Dict[str, int],
    Dict[str, Tuple[int, float]]
]
FloatValuedMetric = Union[
    Dict[str, float],
    Dict[str, Tuple[float, float]]
]


class PerCUIMetrics(BaseModel):
    weights: List[Union[int, float]] = []
    vals: List[Union[int, float]] = []

    def add(self, val, weight: int = 1):
        self.weights.append(weight)
        self.vals.append(val)

    def get_mean(self):
        return sum(w * v for w, v in zip(self.weights, self.vals)) / sum(self.weights)

    def get_std(self):
        mean = self.get_mean()
        return (sum(w * (v - mean)**2 for w, v in zip(self.weights, self.vals)) / sum(self.weights))**.5


def _add_helper(joined: List[Dict[str, PerCUIMetrics]],
                single: List[Dict[str, int]]) -> None:
    if len(joined) != len(single):
        raise ValueError(f"Incompatible number of stuff: {len(joined)} vs {len(single)}")
    for j, s in zip(joined, single):
        for k, v in s.items():
            if k not in j:
                j[k] = PerCUIMetrics()
            j[k].add(v)


def _add_weighted_helper(joined: List[Dict[str, PerCUIMetrics]],
                         single: List[Dict[str, float]],
                         cui2count: Dict[str, int]) -> None:
    for j, s in zip(joined, single):
        for k, v in s.items():
            if k not in j:
                j[k] = PerCUIMetrics()
            j[k].add(v, cui2count[k])


def get_metrics_mean(metrics: List[Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]],
                     include_std: bool) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
    """The the mean of the provided metrics.

    Args:
        metrics (List[Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]): The metrics.
        include_std (bool): Whether to include the standard deviation.

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
    all_fps: Dict[str, PerCUIMetrics] = {}
    all_fns: Dict[str, PerCUIMetrics] = {}
    all_tps: Dict[str, PerCUIMetrics] = {}
    # weighted-averages
    all_cui_prec: Dict[str, PerCUIMetrics] = {}
    all_cui_rec: Dict[str, PerCUIMetrics] = {}
    all_cui_f1: Dict[str, PerCUIMetrics] = {}
    # additive
    all_cui_counts: Dict[str, PerCUIMetrics] = {}
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
        cur_adds = list(current[:3]) + [cur_counts]
        _add_helper(all_additives, cur_adds)
        _add_weighted_helper(all_weighted_averages, cur_wa, cur_counts)
        cur_examples = current[-1]
        _merge_examples(all_examples, cur_examples)
    # conversion from PerCUI metrics to int/float and (if needed) STD
    cui_fps: IntValuedMetric = {}
    cui_fns: IntValuedMetric = {}
    cui_tps: IntValuedMetric = {}
    cui_prec: FloatValuedMetric = {}
    cui_rec: FloatValuedMetric = {}
    cui_f1: FloatValuedMetric = {}
    final_counts: IntValuedMetric = {}
    to_change: List[Union[IntValuedMetric, FloatValuedMetric]] = [
        cui_fps, cui_fns, cui_tps, final_counts,
        cui_prec, cui_rec, cui_f1,
    ]
    # get the mean and/or std
    for nr, (df, d) in enumerate(zip(to_change, all_additives + all_weighted_averages)):
        for k, v in d.items():
            if nr == 3 and not include_std:
                # counts need to be added up
                # NOTE: the type:ignore comment _shouldn't_ be necessary
                #       but mypy thinks we're setting a float or integer
                #       where a tuple is expected
                df[k] = sum(v.vals)  # type: ignore
                # NOTE: The current implementation shows the sum for counts
                #       if not STD is required, but the mean along with the
                #       standard deviation if the latter is required.
            elif not include_std:
                df[k] = v.get_mean()
            else:
                # NOTE: the type:ignore comment _shouldn't_ be necessary
                #       but mypy thinks we're setting a tuple
                #       where a float or integer is expected
                df[k] = (v.get_mean(), v.get_std())  # type: ignore
    return (cui_fps, cui_fns, cui_tps, cui_prec, cui_rec, cui_f1,
            final_counts, all_examples)


def get_k_fold_stats(cat: CATLike, mct_export_data: MedCATTrainerExport, k: int = 3,
                     split_type: SplitType = SplitType.DOCUMENTS_WEIGHTED,
                     include_std: bool = False, *args, **kwargs) -> Tuple:
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
        split_type (SplitType): Whether to use annodations or docs. Defaults to DOCUMENTS_WEIGHTED.
        include_std (bool): Whether to include stanrdard deviation. Defaults to False.
        *args: Arguments passed to the `CAT.train_supervised_raw` method.
        **kwargs: Keyword arguments passed to the `CAT.train_supervised_raw` method.

    Returns:
        Tuple: The averaged metrics. Potentially with their corresponding standard deviations.
    """
    creator = get_fold_creator(mct_export_data, k, split_type=split_type)
    folds = creator.create_folds()
    per_fold_metrics = get_per_fold_metrics(cat, folds, *args, **kwargs)
    means = get_metrics_mean(per_fold_metrics, include_std)
    return means
