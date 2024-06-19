import os
import json
from typing import Dict, Union, Optional
from copy import deepcopy

from medcat.stats import kfold
from medcat.cat import CAT
from pydantic.error_wrappers import ValidationError as PydanticValidationError

import unittest

from .helpers import MCTExportPydanticModel, nullify_doc_names_proj_ids


class MCTExportTests(unittest.TestCase):
    EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..",
                               "resources", "medcat_trainer_export.json")

    @classmethod
    def setUpClass(cls) -> None:
        with open(cls.EXPORT_PATH) as f:
            cls.mct_export = json.load(f)

    def assertIsMCTExport(self, obj):
        try:
            model = MCTExportPydanticModel(**obj)
        except PydanticValidationError as e:
            raise AssertionError("Not n MCT export") from e
        self.assertIsInstance(model, MCTExportPydanticModel)


class KFoldCreatorTests(MCTExportTests):
    K = 3
    SPLIT_TYPE = kfold.SplitType.DOCUMENTS


    def setUp(self) -> None:
        self.creator = kfold.get_fold_creator(self.mct_export, self.K, split_type=self.SPLIT_TYPE)
        self.folds = self.creator.create_folds()

    def test_folding_does_not_modify_initial_export(self):
        with open(self.EXPORT_PATH) as f:
            export_copy = json.load(f)
        self.assertEqual(export_copy, self.mct_export)

    def test_mct_export_has_correct_format(self):
        self.assertIsMCTExport(self.mct_export)

    def test_folds_have_docs(self):
        for nr, fold in enumerate(self.folds):
            with self.subTest(f"Fold-{nr}"):
                self.assertGreater(kfold.count_all_docs(fold), 0)

    def test_folds_have_anns(self):
        for nr, fold in enumerate(self.folds):
            with self.subTest(f"Fold-{nr}"):
                self.assertGreater(kfold.count_all_annotations(fold), 0)

    def test_folds_are_mct_exports(self):
        for nr, fold in enumerate(self.folds):
            with self.subTest(f"Fold-{nr}"):
                self.assertIsMCTExport(fold)

    def test_gets_correct_number_of_folds(self):
        self.assertEqual(len(self.folds), self.K)

    def test_folds_keep_all_docs(self):
        total_docs = 0
        for fold in self.folds:
            docs = kfold.count_all_docs(fold)
            total_docs += docs
        count_all_once = kfold.count_all_docs(self.mct_export)
        if self.SPLIT_TYPE is kfold.SplitType.ANNOTATIONS:
            # NOTE: This may be greater if split in the middle of a document
            #       because that document may then exist in both folds
            self.assertGreaterEqual(total_docs, count_all_once)
        else:
            self.assertEqual(total_docs, count_all_once)

    def test_folds_keep_all_anns(self):
        total_anns = 0
        for fold in self.folds:
            anns = kfold.count_all_annotations(fold)
            total_anns += anns
        count_all_once = kfold.count_all_annotations(self.mct_export)
        self.assertEqual(total_anns, count_all_once)

    def test_1fold_same_as_orig(self):
        folds = kfold.get_fold_creator(self.mct_export, 1, split_type=self.SPLIT_TYPE).create_folds()
        self.assertEqual(len(folds), 1)
        fold, = folds
        self.assertIsInstance(fold, dict)
        self.assertIsMCTExport(fold)
        self.assertEqual(
            nullify_doc_names_proj_ids(self.mct_export),
            nullify_doc_names_proj_ids(fold),
        )

    def test_has_reasonable_annotations_per_folds(self):
        anns_per_folds = [kfold.count_all_annotations(fold) for fold in self.folds]
        print(f"ANNS per folds:\n{anns_per_folds}")
        docs_per_folds = [kfold.count_all_docs(fold) for fold in self.folds]
        print(f"DOCS per folds:\n{docs_per_folds}")


# this is a taylor-made export that
# just contains a few "documents"
# with the fake CUIs "annotated"
NEW_EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..",
                               "resources", "medcat_trainer_export_FAKE_CONCEPTS.json")


class KFoldCreatorPerAnnsTests(KFoldCreatorTests):
    SPLIT_TYPE = kfold.SplitType.ANNOTATIONS


class KFoldCreatorPerWeightedDocsTests(KFoldCreatorTests):
    SPLIT_TYPE = kfold.SplitType.DOCUMENTS_WEIGHTED
    # should have a total of 435, so 145 per in ideal world
    # but we'll allow the following deviation
    PERMITTED_MAX_DEVIATION_IN_ANNS = 5

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.total_anns = kfold.count_all_annotations(cls.mct_export)
        cls.expected_anns_per_fold = cls.total_anns // cls.K
        cls.expected_lower_bound = cls.expected_anns_per_fold - cls.PERMITTED_MAX_DEVIATION_IN_ANNS
        cls.expected_upper_bound = cls.expected_anns_per_fold + cls.PERMITTED_MAX_DEVIATION_IN_ANNS

    def test_has_reasonable_annotations_per_folds(self):
        anns_per_folds = [kfold.count_all_annotations(fold) for fold in self.folds]
        for nr, anns in enumerate(anns_per_folds):
            with self.subTest(f"Fold-{nr}"):
                self.assertGreater(anns, self.expected_lower_bound)
                self.assertLess(anns, self.expected_upper_bound)
        # NOTE: as of testing, this will split [146, 145, 144]
        #       whereas regular per-docs split will have [140, 163, 132]


class KFoldCreatorNewExportTests(KFoldCreatorTests):
    EXPORT_PATH = NEW_EXPORT_PATH


class KFoldCreatorNewExportAnnsTests(KFoldCreatorNewExportTests):
    SPLIT_TYPE = kfold.SplitType.ANNOTATIONS


class KFoldCreatorNewExportWeightedDocsTests(KFoldCreatorNewExportTests):
    SPLIT_TYPE = kfold.SplitType.DOCUMENTS_WEIGHTED


class KFoldCATTests(MCTExportTests):
    _names = ['fps', 'fns', 'tps', 'prec', 'rec', 'f1', 'counts', 'examples']
    EXPORT_PATH = NEW_EXPORT_PATH
    CAT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
    TOLERANCE_PLACES = 10  # tolerance of 10 digits

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.cat = CAT.load_model_pack(cls.CAT_PATH)

    def setUp(self) -> None:
        super().setUp()
        self.reg_stats = self.cat._print_stats(self.mct_export, do_print=False)
        # TODO - remove
        self.maxDiff = 4000

    # NOTE: Due to floating point errors, sometimes we may get slightly different results
    def assertDictsAlmostEqual(self, d1: Dict[str, Union[int, float]], d2: Dict[str, Union[int, float]],
                               tolerance_places: Optional[int] = None) -> None:
        self.assertEqual(d1.keys(), d2.keys())
        tol = tolerance_places if tolerance_places is not None else self.TOLERANCE_PLACES
        for k in d1:
            v1, v2 = d1[k], d2[k]
            self.assertAlmostEqual(v1, v2, places=tol)


class KFoldStatsConsistencyTests(KFoldCATTests):

    def test_mct_export_valid(self):
        self.assertIsMCTExport(self.mct_export)

    def test_stats_consistent(self):
        stats = self.cat._print_stats(self.mct_export, do_print=False)
        for name, stats1, stats2 in zip(self._names, self.reg_stats, stats):
            with self.subTest(name):
                # NOTE: These should be EXACTLY equal since there shouldn't be
                #       any different additions and the like
                self.assertEqual(stats1, stats2)


class KFoldMetricsTests(KFoldCATTests):
    SPLIT_TYPE = kfold.SplitType.DOCUMENTS

    def test_metrics_1_fold_same_as_normal(self):
        stats = kfold.get_k_fold_stats(self.cat, self.mct_export, k=1,
                                       split_type=self.SPLIT_TYPE)
        for name, reg, folds1 in zip(self._names, self.reg_stats, stats):
            with self.subTest(name):
                if name != 'examples':
                    # NOTE: These may not be exactly equal due to floating point errors
                    self.assertDictsAlmostEqual(reg, folds1)
                else:
                    self.assertEqual(reg, folds1)


class KFoldPerAnnsMetricsTests(KFoldMetricsTests):
    SPLIT_TYPE = kfold.SplitType.ANNOTATIONS


class KFoldWeightedDocsMetricsTests(KFoldMetricsTests):
    SPLIT_TYPE = kfold.SplitType.DOCUMENTS_WEIGHTED


class KFoldDuplicatedTests(KFoldCATTests):
    COPIES = 3

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.docs_in_orig = kfold.count_all_docs(cls.mct_export)
        cls.anns_in_orig = kfold.count_all_annotations(cls.mct_export)
        cls.data_copied: kfold.MedCATTrainerExport = deepcopy(cls.mct_export)
        for project in cls.data_copied['projects']:
            documents_list = project['documents']
            copies = documents_list + [
                {k: v if k != 'name' else f"{v}_cp_{nr}" for k, v in doc.items()} for nr in range(cls.COPIES - 1)
                for doc in documents_list
            ]
            project['documents'] = copies
        cls.docs_in_copy = kfold.count_all_docs(cls.data_copied)
        cls.anns_in_copy = kfold.count_all_annotations(cls.data_copied)
        cls.stats_copied = kfold.get_k_fold_stats(cls.cat, cls.data_copied, k=cls.COPIES)
        cls.stats_copied_2 = kfold.get_k_fold_stats(cls.cat, cls.data_copied, k=cls.COPIES)

    # some stats with real model/data will be e.g 0.99 vs 0.9747
    # so in that case, lower it to 1 or so
    _stats_consistency_tolerance = 8

    def test_stats_consistent(self):
        for name, one, two in zip(self._names, self.stats_copied, self.stats_copied_2):
            with self.subTest(name):
                if name == 'examples':
                    # examples are hard
                    # sometimes they differ by quite a lot
                    for etype in one:
                        ev1, ev2 = one[etype], two[etype]
                        with self.subTest(f"{name}-{etype}"):
                            self.assertEqual(ev1.keys(), ev2.keys())
                            for cui in ev1:
                                per_cui_examples1 = ev1[cui]
                                per_cui_examples2 = ev2[cui]
                                with self.subTest(f"{name}-{etype}-{cui}-[{self.cat.cdb.cui2preferred_name.get(cui, cui)}]"):
                                    self.assertEqual(len(per_cui_examples1), len(per_cui_examples2), "INCORRECT NUMBER OF ITEMS")
                                    for ex1, ex2 in zip(per_cui_examples1, per_cui_examples2):
                                        self.assertDictsAlmostEqual(ex1, ex2, tolerance_places=self._stats_consistency_tolerance)
                    continue
                self.assertEqual(one, two)

    def test_copy_has_correct_number_documents(self):
        self.assertEqual(self.COPIES * self.docs_in_orig, self.docs_in_copy)

    def test_copy_has_correct_number_annotations(self):
        self.assertEqual(self.COPIES * self.anns_in_orig, self.anns_in_copy)

    def test_3_fold_identical_folds(self):
        folds = kfold.get_fold_creator(self.data_copied, nr_of_folds=self.COPIES,
                                  split_type=kfold.SplitType.DOCUMENTS).create_folds()
        self.assertEqual(len(folds), self.COPIES)
        for nr, fold in enumerate(folds):
            with self.subTest(f"Fold-{nr}"):
                # if they're all equal to original, they're eqaul to each other
                self.assertEqual(
                    nullify_doc_names_proj_ids(fold),
                    nullify_doc_names_proj_ids(self.mct_export)
                )

    def test_metrics_3_fold(self):
        stats_simple = self.reg_stats
        for name, old, new in zip(self._names, stats_simple, self.stats_copied):
            if name == 'examples':
                continue
            # with self.subTest(name):
            if name in ("fps", "fns", "tps", "counts"):
                # count should be triples
                pass
            if name in ("prec", "rec", "f1"):
                # these should average to the same ??
                all_keys = old.keys() | new.keys()
                for cui in all_keys:
                    cuiname = self.cat.cdb.cui2preferred_name.get(cui, cui)
                    with self.subTest(f"{name}-{cui} [{cuiname}]"):
                        self.assertIn(cui, old.keys(), f"CUI '{cui}' ({cuiname}) not in old")
                        self.assertIn(cui, new.keys(), f"CUI '{cui}' ({cuiname}) not in new")
                        v1, v2 = old[cui], new[cui]
                        self.assertEqual(v1, v2, f"Values not equal for {cui} ({self.cat.cdb.cui2preferred_name.get(cui, cui)})")
