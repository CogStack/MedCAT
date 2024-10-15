import json
import os
import sys
import time
from typing import Callable
from functools import partial
import unittest
from unittest.mock import mock_open, patch
import tempfile
import shutil
import logging
import contextlib
import humanfriendly
from transformers import AutoTokenizer
from medcat.vocab import Vocab
from medcat.cdb import CDB, logger as cdb_logger
from medcat.cat import CAT, logger as cat_logger
from medcat.config import Config
from medcat.pipe import logger as pipe_logger
from medcat.utils.checkpoint import Checkpoint
from medcat.meta_cat import MetaCAT
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBERT


class CATTests(unittest.TestCase):
    SUPERVISED_TRAINING_JSON = os.path.join(os.path.dirname(__file__), "resources", "medcat_trainer_export.json")

    @classmethod
    def setUpClass(cls) -> None:
        cls.cdb = CDB.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "cdb.dat"))
        cls.vocab = Vocab.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "vocab.dat"))
        cls.vocab.make_unigram_table()
        cls.cdb.config.general.spacy_model = "en_core_web_md"
        cls.cdb.config.ner.min_name_len = 2
        cls.cdb.config.ner.upper_case_limit_len = 3
        cls.cdb.config.general.spell_check = True
        cls.cdb.config.linking.train_count_threshold = 10
        cls.cdb.config.linking.similarity_threshold = 0.3
        cls.cdb.config.linking.train = True
        cls.cdb.config.linking.disamb_length_limit = 5
        cls.cdb.config.general.full_unlink = True
        cls._temp_logs_folder = tempfile.TemporaryDirectory()
        cls.cdb.config.general.usage_monitor.enabled = True
        cls.cdb.config.general.usage_monitor.log_folder = cls._temp_logs_folder.name
        cls.meta_cat_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        cls.undertest = CAT(cdb=cls.cdb, config=cls.cdb.config, vocab=cls.vocab, meta_cats=[])
        cls._linkng_filters = cls.undertest.config.linking.filters.copy_of()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.undertest.destroy_pipe()
        if os.path.exists(cls.meta_cat_dir):
            shutil.rmtree(cls.meta_cat_dir)
        cls._temp_logs_folder.cleanup()

    def setUp(self):
        self._temp_file = tempfile.NamedTemporaryFile()
        self.cdb.config.general.simple_hash = False

    def tearDown(self) -> None:
        self.cdb.config.annotation_output.include_text_in_output = False
        # need to make sure linking filters are not retained beyond a test scope
        self.undertest.config.linking.filters = self._linkng_filters.copy_of()
        self._temp_file.close()
        # remove existing contents / empty file log file
        log_file_path = self.undertest.usage_monitor.log_file
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

    def test_callable_with_single_text(self):
        text = "The dog is sitting outside the house."
        doc = self.undertest(text)
        self.assertEqual(text, doc.text)

    def test_callable_with_single_empty_text(self):
        self.assertIsNone(self.undertest(""))

    def test_callable_with_single_none_text(self):
        self.assertIsNone(self.undertest(None))

    in_data_mp = [
        (1, "The dog is sitting outside the house and second csv."),
        (2, ""),
        (3, None)
    ]

    def test_multiprocessing(self):
        self.assert_mp_works(self.in_data_mp)

    def assert_mp_works(self, in_data, **kwargs):
        out = self.undertest.multiprocessing_batch_char_size(in_data, nproc=1, **kwargs)

        self.assertEqual(3, len(out))
        self.assertEqual(1, len(out[1]['entities']))
        self.assertEqual(0, len(out[2]['entities']))
        self.assertEqual(0, len(out[3]['entities']))

    def test_multiprocessing_with_generator(self):
        # NOTE: generators won't have full use of 
        #       the same progress bar functionality
        #       but we're still hoping they would work in general
        in_generator = (part for part in self.in_data_mp)
        self.assert_mp_works(in_generator)

    def test_multiprocessing_works_min_memory_size(self):
        self.assert_mp_works(self.in_data_mp, min_free_memory_size="1GB")

    def test_mp_fails_incorrect_min_mem(self):
        in_data = [(nr, f"nr:{nr}") for nr in range(4)]
        with self.assertRaises(humanfriendly.InvalidSize):
            self.undertest.multiprocessing_batch_char_size(in_data, nproc=2, batch_size_chars=10,
                                                           min_free_memory_size="100nm")

    def test_mp_fails_both_min_mem(self):
        in_data = [(nr, f"nr:{nr}") for nr in range(4)]
        with self.assertRaises(ValueError):
            self.undertest.multiprocessing_batch_char_size(in_data, nproc=2, batch_size_chars=10,
                                                           min_free_memory_size="10GB",
                                                           min_free_memory=0.20)


    @contextlib.contextmanager
    def _assert_logs_in_temp_file(self, logger: logging.Logger):
        # NOTE: The reason I need to do this is because multiprocessing is used
        #       and because of that I can't use the assertLogs method
        #       because the in different threads difference instances are used.
        #       however, if I force to use a file, it should still save on disk
        handler = logging.FileHandler(self._temp_file.name)
        logger.addHandler(handler)
        contents_before = self._temp_file.read()
        yield
        logger.removeHandler(handler)
        contents_after = self._temp_file.read()
        self.assertNotEqual(contents_before, contents_after)

    def test_mp_logs_failure_all_min_mem(self):
        in_data = [(nr, f"nr:{nr}") for nr in range(4)]
        with self._assert_logs_in_temp_file(cat_logger):
            self.undertest.multiprocessing_batch_char_size(in_data, nproc=2, batch_size_chars=10,
                                                           min_free_memory_size="100PB")

    def test_mp_logs_failure_min_mem_fraction(self):
        in_data = [(nr, f"nr:{nr}") for nr in range(4)]
        with self._assert_logs_in_temp_file(cat_logger):
            self.undertest.multiprocessing_batch_char_size(in_data, nproc=2, batch_size_chars=10,
                                                           min_free_memory=1.0)

    def test_multiprocessing_pipe(self):
        in_data = [
            (1, "The dog is sitting outside the house and second csv."),
            (2, "The dog is sitting outside the house."),
            (3, "The dog is sitting outside the house."),
        ]
        out = self.undertest.multiprocessing_batch_docs_size(in_data, nproc=2, return_dict=False)
        self.assertTrue(type(out) is list)
        self.assertEqual(3, len(out))
        self.assertEqual(1, out[0][0])
        self.assertEqual('second csv', out[0][1]['entities'][0]['source_value'])
        self.assertEqual(2, out[1][0])
        self.assertEqual({'entities': {}, 'tokens': []}, out[1][1])
        self.assertEqual(3, out[2][0])
        self.assertEqual({'entities': {}, 'tokens': []}, out[2][1])

    def test_multiprocessing_pipe_with_malformed_texts(self):
        in_data = [
            (1, "The dog is sitting outside the house."),
            (2, ""),
            (3, None),
        ]
        out = self.undertest.multiprocessing_batch_docs_size(in_data, nproc=1, batch_size=1, return_dict=False)
        self.assertTrue(type(out) is list)
        self.assertEqual(3, len(out))
        self.assertEqual(1, out[0][0])
        self.assertEqual({'entities': {}, 'tokens': []}, out[0][1])
        self.assertEqual(2, out[1][0])
        self.assertEqual({'entities': {}, 'tokens': []}, out[1][1])
        self.assertEqual(3, out[2][0])
        self.assertEqual({'entities': {}, 'tokens': []}, out[2][1])

    def test_multiprocessing_pipe_return_dict(self):
        in_data = [
            (1, "The dog is sitting outside the house."),
            (2, "The dog is sitting outside the house."),
            (3, "The dog is sitting outside the house.")
        ]
        out = self.undertest.multiprocessing_batch_docs_size(in_data, nproc=2, return_dict=True)
        self.assertTrue(type(out) is dict)
        self.assertEqual(3, len(out))
        self.assertEqual({'entities': {}, 'tokens': []}, out[1])
        self.assertEqual({'entities': {}, 'tokens': []}, out[2])
        self.assertEqual({'entities': {}, 'tokens': []}, out[3])

    def test_train(self):
        with tempfile.TemporaryDirectory() as temp_file:
            self._test_train(temp_file)

    def _test_train(self, temp_file: str):
        ckpt_steps = 2
        nepochs = 3
        ckpt_dir_path = temp_file
        checkpoint = Checkpoint(dir_path=ckpt_dir_path, steps=ckpt_steps)
        self.undertest.cdb.print_stats()
        self.undertest.train(["The dog is not a house"] * 20, nepochs=nepochs, checkpoint=checkpoint)
        self.undertest.cdb.print_stats()
        checkpoints = [f for f in os.listdir(ckpt_dir_path) if "checkpoint-" in f]

        self.assertEqual(1, len(checkpoints))
        self.assertEqual(f"checkpoint-{ckpt_steps}-{nepochs * 20}", checkpoints[0])

    def test_resume_training(self):
        with tempfile.TemporaryDirectory() as temp_file:
            self._test_resume_training(temp_file)

    def _test_resume_training(self, temp_file: str):
        nepochs_train = 1
        nepochs_retrain = 1
        ckpt_steps = 3
        ckpt_dir_path = temp_file
        checkpoint = Checkpoint(dir_path=ckpt_dir_path, steps=ckpt_steps, max_to_keep=sys.maxsize)
        self.undertest.cdb.print_stats()
        self.undertest.train(["The dog is not a house"] * 20,
                             nepochs=nepochs_train,
                             checkpoint=checkpoint,
                             is_resumed=False)
        self.undertest.cdb.print_stats()
        self.undertest.train(["The dog is not a house"] * 20,
                             nepochs=nepochs_train+nepochs_retrain,
                             checkpoint=checkpoint,
                             is_resumed=True)
        checkpoints = [f for f in os.listdir(ckpt_dir_path) if "checkpoint-" in f]
        self.assertEqual(13, len(checkpoints))
        self.assertTrue("checkpoint-%s-3" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-6" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-9" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-12" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-15" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-18" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-21" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-24" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-27" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-30" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-33" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-36" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-39" % ckpt_steps in checkpoints)

    def test_resume_training_on_absent_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            self._test_resume_training_on_absent_checkpoints(temp_dir_name)

    def _test_resume_training_on_absent_checkpoints(self, temp_dir_name):
        ckpt_dir_path = temp_dir_name
        checkpoint = Checkpoint(dir_path=ckpt_dir_path)
        with self.assertRaises(Exception) as e:
            self.undertest.train(["The dog is not a house"] * 40, checkpoint=checkpoint, is_resumed=True)
        self.assertEqual("Checkpoints not found. You need to train from scratch.", str(e.exception))

    def test_train_keep_n_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            self._test_train_keep_n_checkpoints(temp_dir_name)

    def _test_train_keep_n_checkpoints(self, temp_dir_name):
        ckpt_steps = 2
        ckpt_dir_path = temp_dir_name
        checkpoint = Checkpoint(dir_path=ckpt_dir_path, steps=ckpt_steps, max_to_keep=2)
        self.undertest.cdb.print_stats()
        self.undertest.train(["The dog is not a house"] * 20, checkpoint=checkpoint)
        self.undertest.cdb.print_stats()
        checkpoints = [f for f in os.listdir(ckpt_dir_path) if "checkpoint-" in f]
        self.assertEqual(2, len(checkpoints))
        self.assertTrue("checkpoint-%s-18" % ckpt_steps in checkpoints)
        self.assertTrue("checkpoint-%s-20" % ckpt_steps in checkpoints)

    def test_get_entities(self):
        text = "The dog is sitting outside the house."
        out = self.undertest.get_entities(text)
        self.assertEqual({}, out["entities"])
        self.assertEqual([], out["tokens"])
        self.assertFalse("text" in out)

    def test_get_entities_including_text(self):
        self.cdb.config.annotation_output.include_text_in_output = True
        text = "The dog is sitting outside the house."
        out = self.undertest.get_entities(text)
        self.assertEqual({}, out["entities"])
        self.assertEqual([], out["tokens"])
        self.assertTrue(text in out["text"])

    def test_get_entities_multi_texts(self):
        in_data = [(1, "The dog is sitting outside the house."), (2, ""), (3, "The dog is sitting outside the house.")]
        out = self.undertest.get_entities_multi_texts(in_data, n_process=2)
        self.assertEqual(3, len(out))
        self.assertFalse("text" in out[0])
        self.assertFalse("text" in out[1])
        self.assertFalse("text" in out[2])

    def test_get_entities_multi_texts_including_text(self):
        self.cdb.config.annotation_output.include_text_in_output = True
        in_data = [(1, "The dog is sitting outside the house."), (2, ""), (3, None)]
        out = self.undertest.get_entities_multi_texts(in_data, n_process=2)
        self.assertEqual(3, len(out))
        self.assertTrue("text" in out[0])
        self.assertFalse("text" in out[1])
        self.assertFalse("text" in out[2])

    def test_train_supervised(self):
        with tempfile.TemporaryDirectory() as temp_file:
            self._test_train_superivsed(temp_file)

    def _test_train_superivsed(self, temp_file: str):
        nepochs = 3
        num_of_documents = 27
        data_path = self.SUPERVISED_TRAINING_JSON
        ckpt_dir_path = temp_file
        checkpoint = Checkpoint(dir_path=ckpt_dir_path, steps=1, max_to_keep=sys.maxsize)
        fp, fn, tp, p, r, f1, cui_counts, examples = self.undertest.train_supervised_from_json(data_path,
                                                                                               checkpoint=checkpoint,
                                                                                               nepochs=nepochs)
        checkpoints = [f for f in os.listdir(ckpt_dir_path) if "checkpoint-" in f]
        self.assertEqual({}, fp)
        self.assertEqual({}, fn)
        self.assertEqual({}, tp)
        self.assertEqual({}, p)
        self.assertEqual({}, r)
        self.assertEqual({}, f1)
        self.assertEqual({}, cui_counts)
        self.assertEqual({}, examples)
        self.assertEqual(nepochs * num_of_documents, len(checkpoints))
        for step in range(1, nepochs * num_of_documents + 1):
            self.assertTrue(f"checkpoint-1-{step}" in checkpoints)

    def test_resume_supervised_training(self):
        with tempfile.TemporaryDirectory() as temp_file:
            self._test_resume_supervised_training(temp_file)

    def _test_resume_supervised_training(self, temp_file: str):
        nepochs_train = 1
        nepochs_retrain = 2
        num_of_documents = 27
        data_path = os.path.join(os.path.dirname(__file__), "resources", "medcat_trainer_export.json")
        ckpt_dir_path = temp_file
        checkpoint = Checkpoint(dir_path=ckpt_dir_path, steps=1, max_to_keep=sys.maxsize)
        self.undertest.train_supervised_from_json(data_path,
                                                  checkpoint=checkpoint,
                                                  nepochs=nepochs_train)
        fp, fn, tp, p, r, f1, cui_counts, examples = self.undertest.train_supervised_from_json(
            data_path, checkpoint=checkpoint, nepochs=nepochs_train+nepochs_retrain, is_resumed=True)
        checkpoints = [f for f in os.listdir(ckpt_dir_path) if "checkpoint-" in f]
        self.assertEqual({}, fp)
        self.assertEqual({}, fn)
        self.assertEqual({}, tp)
        self.assertEqual({}, p)
        self.assertEqual({}, r)
        self.assertEqual({}, f1)
        self.assertEqual({}, cui_counts)
        self.assertEqual({}, examples)
        self.assertEqual((nepochs_train + nepochs_retrain) * num_of_documents, len(checkpoints))
        for step in range(1, (nepochs_train + nepochs_retrain) * num_of_documents):
            self.assertTrue(f"checkpoint-1-{step}" in checkpoints)

    def test_train_supervised_does_not_retain_MCT_filters_default(self, extra_cui_filter=None):
        data_path = os.path.join(os.path.dirname(__file__), "resources", "medcat_trainer_export_filtered.json")
        before = str(self.undertest.config.linking.filters)
        self.undertest.train_supervised_from_json(data_path, nepochs=1, use_filters=True, extra_cui_filter=extra_cui_filter)
        after = str(self.undertest.config.linking.filters)
        self.assertEqual(before, after)

    def test_train_supervised_can_retain_MCT_filters(self, extra_cui_filter=None, retain_extra_cui_filter=False):
        data_path = os.path.join(os.path.dirname(__file__), "resources", "medcat_trainer_export_filtered.json")
        before = str(self.undertest.config.linking.filters)
        self.undertest.train_supervised_from_json(data_path, nepochs=1, use_filters=True, retain_filters=True,
                                                  extra_cui_filter=extra_cui_filter, retain_extra_cui_filter=retain_extra_cui_filter)
        after = str(self.undertest.config.linking.filters)
        self.assertNotEqual(before, after)
        with open(data_path, 'r') as f:
            project0 = json.load(f)['projects'][0]
        filtered_cuis = project0['cuis'].split(',')
        if extra_cui_filter and retain_extra_cui_filter:
            # in case of extra_cui_filter and its retention, only it is retained
            filtered_cuis = extra_cui_filter
        self.assertGreater(len(filtered_cuis), 0)
        for filtered_cui in filtered_cuis:
            with self.subTest(f'CUI: {filtered_cui}'):
                self.assertTrue(filtered_cui in self.undertest.config.linking.filters.cuis)

    def test_train_supervised_no_leak_extra_cui_filters(self):
        self.test_train_supervised_does_not_retain_MCT_filters_default(extra_cui_filter={'C123', 'C111'})

    def test_train_supervised_no_leak_extra_cui_filters_along_MCT_filters(self):
        self.test_train_supervised_can_retain_MCT_filters(extra_cui_filter={'C0037284'})

    def test_train_supervised_can_retain_extra_cui_filters_along_MCT_filters(self):
        self.test_train_supervised_can_retain_MCT_filters(extra_cui_filter={'C0037284'}, retain_extra_cui_filter=True)

    def test_no_error_handling_on_none_input(self):
        out = self.undertest.get_entities(None)
        self.assertEqual({}, out["entities"])
        self.assertEqual([], out["tokens"])

    def test_no_error_handling_on_empty_string_input(self):
        out = self.undertest.get_entities("")
        self.assertEqual({}, out["entities"])
        self.assertEqual([], out["tokens"])

    def test_no_raise_on_single_process_with_none(self):
        out = self.undertest.get_entities_multi_texts(["The dog is sitting outside the house.", None, "The dog is sitting outside the house."], n_process=1, batch_size=2)
        self.assertEqual(3, len(out))
        self.assertEqual({}, out[0]["entities"])
        self.assertEqual([], out[0]["tokens"])
        self.assertEqual({}, out[1]["entities"])
        self.assertEqual([], out[1]["tokens"])
        self.assertEqual({}, out[2]["entities"])
        self.assertEqual([], out[2]["tokens"])

    def test_no_raise_on_single_process_with_empty_string(self):
        out = self.undertest.get_entities_multi_texts(["The dog is sitting outside the house.", "", "The dog is sitting outside the house."], n_process=1, batch_size=2)
        self.assertEqual(3, len(out))
        self.assertEqual({}, out[0]["entities"])
        self.assertEqual([], out[0]["tokens"])
        self.assertEqual({}, out[1]["entities"])
        self.assertEqual([], out[1]["tokens"])
        self.assertEqual({}, out[2]["entities"])
        self.assertEqual([], out[2]["tokens"])

    def test_error_handling_multi_processes(self):
        self.cdb.config.annotation_output.include_text_in_output = True
        out = self.undertest.get_entities_multi_texts([
                                           (1, "The dog is sitting outside the house 1."),
                                           (2, "The dog is sitting outside the house 2."),
                                           (3, "The dog is sitting outside the house 3."),
                                           (4, None),
                                           (5, None)], n_process=2, batch_size=2)
        self.assertEqual(5, len(out))
        self.assertEqual({}, out[0]["entities"])
        self.assertEqual([], out[0]["tokens"])
        self.assertTrue("The dog is sitting outside the house 1.", out[0]["text"])
        self.assertEqual({}, out[1]["entities"])
        self.assertEqual([], out[1]["tokens"])
        self.assertTrue("The dog is sitting outside the house 2.", out[1]["text"])
        self.assertEqual({}, out[2]["entities"])
        self.assertEqual([], out[2]["tokens"])
        self.assertTrue("The dog is sitting outside the house 3.", out[2]["text"])
        self.assertEqual({}, out[3]["entities"])
        self.assertEqual([], out[3]["tokens"])
        self.assertFalse("text" in out[3])
        self.assertEqual({}, out[4]["entities"])
        self.assertEqual([], out[4]["tokens"])
        self.assertFalse("text" in out[4])

    def test_create_model_pack(self):
        with tempfile.TemporaryDirectory() as save_dir_path:
            self._test_create_model_pack(save_dir_path)

    def _test_create_model_pack(self, save_dir_path):
        cat = CAT(cdb=self.cdb, config=self.cdb.config, vocab=self.vocab, meta_cats=[_get_meta_cat(self.meta_cat_dir)])
        full_model_pack_name = cat.create_model_pack(save_dir_path, model_pack_name="mp_name")
        pack = [f for f in os.listdir(save_dir_path)]
        self.assertTrue(full_model_pack_name in pack)
        self.assertTrue(f'{full_model_pack_name}.zip' in pack)
        contents = [f for f in os.listdir(os.path.join(save_dir_path, full_model_pack_name))]
        self.assertTrue("cdb.dat" in contents)
        self.assertTrue("vocab.dat" in contents)
        self.assertTrue("model_card.json" in contents)
        self.assertTrue("meta_Status" in contents)
        with open(os.path.join(save_dir_path, full_model_pack_name, "model_card.json")) as file:
            model_card = json.load(file)
        self.assertTrue("MedCAT Version" in model_card)

    def test_load_model_pack(self):
        with tempfile.TemporaryDirectory() as save_dir_path:
            self._test_load_model_pack(save_dir_path)

    def _test_load_model_pack(self, save_dir_path):
        meta_cat = _get_meta_cat(self.meta_cat_dir)
        cat = CAT(cdb=self.cdb, config=self.cdb.config, vocab=self.vocab, meta_cats=[meta_cat])
        full_model_pack_name = cat.create_model_pack(save_dir_path, model_pack_name="mp_name")
        cat = CAT.load_model_pack(os.path.join(save_dir_path, f"{full_model_pack_name}.zip"))
        self.assertTrue(isinstance(cat, CAT))
        self.assertIsNotNone(cat.config.version.medcat_version)
        self.assertEqual(repr(cat._meta_cats), repr([meta_cat]))

    def test_load_model_pack_without_meta_cat(self):
        with tempfile.TemporaryDirectory() as save_dir_path:
            self._test_load_model_pack_without_meta_cat(save_dir_path)

    def _test_load_model_pack_without_meta_cat(self, save_dir_path):
        meta_cat = _get_meta_cat(self.meta_cat_dir)
        cat = CAT(cdb=self.cdb, config=self.cdb.config, vocab=self.vocab, meta_cats=[meta_cat])
        full_model_pack_name = cat.create_model_pack(save_dir_path, model_pack_name="mp_name")
        cat = CAT.load_model_pack(os.path.join(save_dir_path, f"{full_model_pack_name}.zip"), load_meta_models=False)
        self.assertTrue(isinstance(cat, CAT))
        self.assertIsNotNone(cat.config.version.medcat_version)
        self.assertEqual(cat._meta_cats, [])

    def test_hashing(self):
        with tempfile.TemporaryDirectory() as save_dir_path:
            self._test_hashing(save_dir_path)

    def _test_hashing(self, save_dir_path):
        full_model_pack_name = self.undertest.create_model_pack(save_dir_path, model_pack_name="mp_name")
        cat = CAT.load_model_pack(os.path.join(save_dir_path, f"{full_model_pack_name}.zip"))
        self.assertEqual(cat.get_hash(), cat.config.version.id)

    def test_print_stats(self):
        # based on current JSON
        EXP_FALSE_NEGATIVES = {'C0017168': 2, 'C0020538': 43, 'C0038454': 4, 'C0007787': 1, 'C0155626': 4, 'C0011860': 12,
                               'C0042029': 6, 'C0010068': 2, 'C0007222': 1, 'C0027051': 6, 'C0878544': 1, 'C0020473': 12,
                               'C0037284': 21, 'C0003864': 4, 'C0011849': 12, 'C0005686': 1, 'C0085762': 3, 'C0030920': 2,
                               'C0854135': 3, 'C0004096': 4, 'C0010054': 10, 'C0497156': 10, 'C0011334': 2, 'C0018939': 1,
                               'C1561826': 2, 'C0276289': 2, 'C0041834': 9, 'C0000833': 2, 'C0238792': 1, 'C0040034': 3,
                               'C0035078': 5, 'C0018799': 5, 'C0042109': 1, 'C0035439': 1, 'C0035435': 1, 'C0018099': 1,
                               'C1277187': 1, 'C0024117': 7, 'C0004238': 4, 'C0032227': 6, 'C0008679': 1, 'C0013146': 6,
                               'C0032285': 1, 'C0002871': 7, 'C0149871': 4, 'C0442886': 1, 'C0022104': 1, 'C0034065': 5,
                               'C0011854': 6, 'C1398668': 1, 'C0020676': 2, 'C1301700': 1, 'C0021167': 1, 'C0029456': 2,
                               'C0011570': 10, 'C0009324': 1, 'C0011882': 1, 'C0020615': 1, 'C0242510': 2, 'C0033581': 2,
                               'C0011168': 3, 'C0039082': 2, 'C0009241': 2, 'C1404970': 1, 'C0018524': 3, 'C0150063': 1,
                               'C0917799': 1, 'C0178417': 1, 'C0033975': 1, 'C0011253': 1, 'C0018802': 8, 'C0022661': 4,
                               'C0017658': 1, 'C0023895': 2, 'C0003123': 1, 'C0041582': 4, 'C0085096': 4, 'C0403447': 2,
                               'C2363741': 2, 'C0457949': 1, 'C0040336': 1, 'C0037315': 2, 'C0024236': 3, 'C0442874': 1,
                               'C0028754': 4, 'C0520679': 5, 'C0028756': 2, 'C0029408': 5, 'C0409959': 2, 'C0018801': 1, 
                               'C3844825': 1, 'C0022660': 2, 'C0005779': 4, 'C0011175': 1, 'C0018965': 4, 'C0018889': 1,
                               'C0022354': 2, 'C0033377': 1, 'C0042769': 1, 'C0035222': 1, 'C1456868': 2, 'C1145670': 1,
                               'C0018790': 1, 'C0263746': 1, 'C0206172': 1, 'C0021400': 1, 'C0243026': 1, 'C0020443': 1,
                               'C0001883': 1, 'C0031350': 1, 'C0010709': 4, 'C1565489': 7, 'C3489393': 1, 'C0005586': 2,
                               'C0158288': 5, 'C0700594': 4, 'C0158266': 3, 'C0006444': 2, 'C0024003': 1}
        with open(self.SUPERVISED_TRAINING_JSON) as f:
            data = json.load(f)
        (fps, fns, tps,
         cui_prec, cui_rec, cui_f1,
         cui_counts, examples) = self.undertest._print_stats(data)
        self.assertEqual(fps, {})
        self.assertEqual(fns, EXP_FALSE_NEGATIVES)
        self.assertEqual(tps, {})
        self.assertEqual(cui_prec, {})
        self.assertEqual(cui_rec, {})
        self.assertEqual(cui_f1, {})
        self.assertEqual(len(cui_counts), 136)
        self.assertEqual(len(examples), 3)

    def _assertNoLogs(self, logger: logging.Logger, level: int):
        if hasattr(self, 'assertNoLogs'):
            return self.assertNoLogs(logger=logger, level=level)
        else:
            return self.__assertNoLogs(logger=logger, level=level)

    @contextlib.contextmanager
    def __assertNoLogs(self, logger: logging.Logger, level: int):
        try:
            with self.assertLogs(logger, level) as captured_logs:
                yield
        except AssertionError:
            return
        if captured_logs:
            raise AssertionError("Logs were found: {}".format(captured_logs))

    def assertLogsDuringAddAndTrainConcept(self, logger: logging.Logger, log_level,
                                           name: str, name_status: str, nr_of_calls: int):
        cui = 'CUI-%d' % (hash(name) + id(name))
        with (self.assertLogs(logger=logger, level=log_level)
              if nr_of_calls == 1
              else self._assertNoLogs(logger=logger, level=log_level)):
            self.undertest.add_and_train_concept(cui, name, name_status=name_status)

    def test_add_and_train_concept_cat_nowarn_long_name(self):
        long_name = 'a very long name'
        self.assertLogsDuringAddAndTrainConcept(cat_logger, logging.WARNING, name=long_name, name_status='', nr_of_calls=0)

    def test_add_and_train_concept_cdb_nowarn_long_name(self):
        long_name = 'a very long name'
        self.assertLogsDuringAddAndTrainConcept(cdb_logger, logging.WARNING, name=long_name, name_status='', nr_of_calls=0)

    def test_add_and_train_concept_cat_nowarn_short_name_not_pref(self):
        short_name = 'a'
        self.assertLogsDuringAddAndTrainConcept(cat_logger, logging.WARNING, name=short_name, name_status='', nr_of_calls=0)

    def test_add_and_train_concept_cdb_nowarn_short_name_not_pref(self):
        short_name = 'a'
        self.assertLogsDuringAddAndTrainConcept(cdb_logger, logging.WARNING, name=short_name, name_status='', nr_of_calls=0)

    def test_add_and_train_concept_cat_warns_short_name(self):
        short_name = 'a'
        self.assertLogsDuringAddAndTrainConcept(cat_logger, logging.WARNING, name=short_name, name_status='P', nr_of_calls=1)

    def test_add_and_train_concept_cdb_warns_short_name(self):
        short_name = 'a'
        self.assertLogsDuringAddAndTrainConcept(cdb_logger, logging.WARNING, name=short_name, name_status='P', nr_of_calls=1)

    def test_get_entities_gets_monitored(self,
                                         text="Some text"):
        repeats = self.undertest.config.general.usage_monitor.batch_size
        # ensure something gets written to the file
        for _ in range(repeats):
            self.undertest.get_entities(text)
        log_file_path = self.undertest.usage_monitor.log_file
        self.assertTrue(os.path.exists(log_file_path))
        with open(log_file_path) as f:
            contents = f.readline()
        self.assertTrue(contents)

    def assert_gets_usage_monitored(self, data_processor: Callable[[None], None], exp_logs: int = 1):
        # clear usage monitor buffer
        self.undertest.usage_monitor.log_buffer.clear()
        data_processor()
        file = self.undertest.usage_monitor.log_file
        if os.path.exists(file):
            with open(file) as f:
                content = f.readlines()
            content += self.undertest.usage_monitor.log_buffer
        else:
            content = self.undertest.usage_monitor.log_buffer
        self.assertTrue(content)
        self.assertEqual(len(content), exp_logs)

    def test_get_entities_logs_usage(self,
                                     text="The dog is sitting outside the house."):
        # clear usage monitor buffer
        self.assert_gets_usage_monitored(partial(self.undertest.get_entities, text), 1)
        line = self.undertest.usage_monitor.log_buffer[0]
        # the 1st element is the input text length
        input_text_length = line.split(",")[1]
        self.assertEqual(str(len(text)), input_text_length)

    TEXT4MP_USAGE = [
        ("ID1", "Text with house and dog one"),
        ("ID2", "Text with house and dog two"),
        ("ID3", "Text with house and dog three"),
        ("ID4", "Text with house and dog four"),
        ("ID5", "Text with house and dog five"),
        ("ID6", "Text with house and dog siz"),
        ("ID7", "Text with house and dog seven"),
        ("ID8", "Text with house and dog eight"),
        ]

    def test_mp_batch_char_size_logs_usage(self):
        all_text = self.TEXT4MP_USAGE
        proc = partial(self.undertest.multiprocessing_batch_char_size, all_text, nproc=2)
        self.assert_gets_usage_monitored(proc, len(all_text))

    def test_mp_get_multi_texts_logs_usage(self):
        all_text = self.TEXT4MP_USAGE
        proc = partial(self.undertest.get_entities_multi_texts, all_text, n_process=2)
        self.assert_gets_usage_monitored(proc, len(all_text))

    def test_mp_batch_docs_size_logs_usage(self):
        all_text = self.TEXT4MP_USAGE
        proc = partial(self.undertest.multiprocessing_batch_docs_size, all_text, nproc=2)
        self.assert_gets_usage_monitored(proc, len(all_text))

    def test_simple_hashing_is_faster(self):
        self.undertest.config.general.simple_hash = False
        st = time.perf_counter()
        self.undertest.get_hash(force_recalc=True)
        took_normal = time.perf_counter() - st
        self.undertest.config.general.simple_hash = True  # will be reset at setUp
        st = time.perf_counter()
        self.undertest.get_hash(force_recalc=True)
        took_simple = time.perf_counter() - st
        # NOTE: In reality simple has should take less than 5 ms
        self.assertLess(took_simple, took_normal)

    def perform_fake_save(self, fake_folder: str = "FAKE_FOLDER"):
        with patch('os.makedirs'):
            with patch('os.path.join', return_value=f"{fake_folder}/data.dat"):
                with patch('builtins.open', mock_open()):
                    with patch('shutil.copytree'):
                        with patch('shutil.make_archive'):
                            # to fix envsnapshot
                            with patch('platform.platform', return_value='TEST'):
                                self.undertest.create_model_pack(fake_folder)
                                self.undertest.config.version.history.append(self.undertest.get_hash())

    def test_subsequent_simple_hashes_same(self):
        self.undertest.config.general.simple_hash = True  # will be reset at setUp
        hash1 = self.undertest.get_hash(force_recalc=True)
        hash2 = self.undertest.get_hash(force_recalc=True)
        self.assertEqual(hash1, hash2)

    def test_simple_hashing_changes_after_save(self):
        self.undertest.config.general.simple_hash = True  # will be reset at setUp
        hash1 = self.undertest.get_hash(force_recalc=True)
        # simulating save
        self.perform_fake_save()
        hash2 = self.undertest.get_hash(force_recalc=True)
        self.assertNotEqual(hash1, hash2)


class GetEntitiesWithStopWords(unittest.TestCase):
    # NB! The order in which the different CDBs are created
    # is important here since the way that the stop words are
    # set is class-based, it creates the side effect of having
    # the same stop words the next time around
    # regardless of whether or not they should've been set

    @classmethod
    def setUpClass(cls) -> None:
        cls.cdb1 = CDB.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "cdb.dat"))
        cls.cdb2 = CDB.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "cdb.dat"))
        cls.vocab = Vocab.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "vocab.dat"))
        cls.vocab.make_unigram_table()
        cls.cdb1.config.general.spacy_model = "en_core_web_md"
        cls.cdb1.config.ner.min_name_len = 2
        cls.cdb1.config.ner.upper_case_limit_len = 3
        cls.cdb1.config.general.spell_check = True
        cls.cdb1.config.linking.train_count_threshold = 10
        cls.cdb1.config.linking.similarity_threshold = 0.3
        cls.cdb1.config.linking.train = True
        cls.cdb1.config.linking.disamb_length_limit = 5
        cls.cdb1.config.general.full_unlink = True
        cls.cdb2.config = Config.from_dict(cls.cdb1.config.asdict())
        # the regular CAT without stopwords
        cls.no_stopwords = CAT(cdb=cls.cdb1, config=cls.cdb1.config, vocab=cls.vocab, meta_cats=[])
        # this (the following two lines)
        # needs to be done before initialising the CAT
        # since that initialises the pipe
        cls.cdb2.config.preprocessing.stopwords = {"stop", "words"}
        cls.cdb2.config.preprocessing.skip_stopwords = True
        # the CAT that skips the stopwords
        cls.w_stopwords = CAT(cdb=cls.cdb2, config=cls.cdb2.config, vocab=cls.vocab, meta_cats=[])

    def test_stopwords_are_skipped(self, text: str = "second words csv"):
        # without stopwords no entities are captured
        # with stopwords, the `second words csv` entity is captured
        doc_no_stopwords = self.no_stopwords(text)
        doc_w_stopwords = self.w_stopwords(text)
        self.assertGreater(len(doc_w_stopwords._.ents), len(doc_no_stopwords._.ents))


class ModelWithTwoConfigsLoadTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples")
        cdb = CDB.load(os.path.join(cls.model_path, "cdb.dat"))
        # save config next to the CDB
        cls.config_path = os.path.join(cls.model_path, 'config.json')
        cdb.config.save(cls.config_path)


    @classmethod
    def tearDownClass(cls) -> None:
        # REMOVE config next to the CDB
        os.remove(cls.config_path)

    def test_loading_model_pack_with_cdb_config_and_config_json_raises_exception(self):
        with self.assertRaises(ValueError):
            CAT.load_model_pack(self.model_path)


class ModelLoadsUnreadableSpacy(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples")
        cdb = CDB.load(os.path.join(model_path, 'cdb.dat'))
        cdb.config.general.spacy_model = os.path.join(cls.temp_dir.name, "en_core_web_md")
        # save CDB in new location
        cdb.save(os.path.join(cls.temp_dir.name, 'cdb.dat'))
        # save config in new location
        cdb.config.save(os.path.join(cls.temp_dir.name, 'config.json'))
        # copy vocab into new location
        vocab_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "vocab.dat")
        cls.vocab_path = os.path.join(cls.temp_dir.name, 'vocab.dat')
        shutil.copyfile(vocab_path, cls.vocab_path)

    @classmethod
    def tearDownClass(cls) -> None:
        # REMOVE temp dir
        cls.temp_dir.cleanup()

    def test_loads_without_specified_spacy_model(self):
        with self.assertLogs(logger=pipe_logger, level=logging.WARNING):
            cat = CAT.load_model_pack(self.temp_dir.name)
        self.assertTrue(isinstance(cat, CAT))


class ModelWithZeroConfigsLoadTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cdb_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "cdb.dat")
        cdb = CDB.load(cdb_path)
        vocab_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "vocab.dat")
        # copy the CDB and vocab to a temp dir
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.cdb_path = os.path.join(cls.temp_dir.name, 'cdb.dat')
        cdb.save(cls.cdb_path) # save without internal config
        cls.vocab_path = os.path.join(cls.temp_dir.name, 'vocab.dat')
        shutil.copyfile(vocab_path, cls.vocab_path)


    @classmethod
    def tearDownClass(cls) -> None:
        # REMOVE temp dir
        cls.temp_dir.cleanup()

    def test_loading_model_pack_without_any_config_raises_exception(self):
        with self.assertRaises(ValueError):
            CAT.load_model_pack(self.temp_dir.name)


def _get_meta_cat(meta_cat_dir):
    config = ConfigMetaCAT()
    config.general["category_name"] = "Status"
    config.train["nepochs"] = 1
    config.model["input_size"] = 10
    meta_cat = MetaCAT(tokenizer=TokenizerWrapperBERT(AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")),
                       embeddings=None,
                       config=config)
    os.makedirs(meta_cat_dir, exist_ok=True)
    json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "mct_export_for_meta_cat_test.json")
    meta_cat.train_from_json(json_path, save_dir_path=meta_cat_dir)
    return meta_cat


class TestLoadingOldWeights(unittest.TestCase):
    cdb_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "examples", "cdb_old_broken_weights_in_config.dat")

    @classmethod
    def setUpClass(cls) -> None:
        cls.cdb = CDB.load(cls.cdb_path)
        cls.wf = cls.cdb.weighted_average_function

    def test_can_call_weights(self):
        res = self.wf(step=1)
        self.assertIsInstance(res, float)


if __name__ == "__main__":
    unittest.main()
