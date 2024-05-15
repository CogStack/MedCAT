import unittest
import os
from unittest import mock
from typing import Callable, Any, Dict
import tempfile

from medcat.utils.cdb_state import captured_state_cdb, CDBState, copy_cdb_state
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.cat import CAT


class StateTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cdb = CDB.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "examples", "cdb.dat"))
        cls.vocab = Vocab.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "examples", "vocab.dat"))
        cls.vocab.make_unigram_table()
        cls.cdb.config.general.spacy_model = "en_core_web_md"
        cls.meta_cat_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        cls.undertest = CAT(cdb=cls.cdb, config=cls.cdb.config, vocab=cls.vocab, meta_cats=[])
        cls.initial_state = copy_cdb_state(cls.cdb)

    @classmethod
    def _set_info(cls, k: str, v: Any, info_dict: Dict):
        info_dict[k] = (len(v), len(str(v)))

    @classmethod
    def do_smth_for_each_state_var(cls, cdb: CDB, callback: Callable[[str, Any], None]) -> None:
        for k in CDBState.__annotations__:
            v = getattr(cdb, k)
            callback(k, v)


class StateSavedTests(StateTests):
    on_disk = False

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # capture state
        with captured_state_cdb(cls.cdb, save_state_to_disk=cls.on_disk):
            # clear state
            cls.do_smth_for_each_state_var(cls.cdb, lambda k, v: v.clear())
            cls.cleared_state = copy_cdb_state(cls.cdb)
        # save after state - should be equal to before
        cls.restored_state = copy_cdb_state(cls.cdb)

    def test_state_saved(self):
        nr_of_targets = len(CDBState.__annotations__)
        self.assertGreater(nr_of_targets, 0)
        self.assertEqual(len(self.initial_state), nr_of_targets)
        self.assertEqual(len(self.cleared_state), nr_of_targets)
        self.assertEqual(len(self.restored_state), nr_of_targets)

    def test_clearing_worked(self):
        self.assertNotEqual(self.initial_state, self.cleared_state)
        for k, v in self.cleared_state.items():
            with self.subTest(k):
                # length is 0
                self.assertFalse(v)

    def test_state_restored(self):
        self.assertEqual(self.initial_state, self.restored_state)


class StateSavedOnDiskTests(StateSavedTests):
    on_disk = True
    _named_tempory_file = tempfile.NamedTemporaryFile

    @classmethod
    def saved_name_temp_file(cls):
        tf = cls._named_tempory_file()
        cls.temp_file_name = tf.name
        return tf

    @classmethod
    def setUpClass(cls) -> None:
        with mock.patch("builtins.open", side_effect=open) as cls.popen:
            with mock.patch("tempfile.NamedTemporaryFile", side_effect=cls.saved_name_temp_file) as cls.pntf:
                return super().setUpClass()

    def test_temp_file_called(self):
        self.pntf.assert_called_once()

    def test_saved_on_disk(self):
        self.popen.assert_called()
        self.assertGreaterEqual(self.popen.call_count, 2)
        self.popen.assert_has_calls([mock.call(self.temp_file_name, 'wb'),
                                     mock.call(self.temp_file_name, 'rb')])


class StateWithTrainingTests(StateTests):
    SUPERVISED_TRAINING_JSON = os.path.join(os.path.dirname(__file__), "..", "resources", "medcat_trainer_export.json")

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        with captured_state_cdb(cls.cdb):
            # do training
            cls.undertest.train_supervised_from_json(cls.SUPERVISED_TRAINING_JSON)
            cls.after_train_state = copy_cdb_state(cls.cdb)
        cls.restored_state = copy_cdb_state(cls.cdb)


class StateRestoredAfterTrain(StateWithTrainingTests):

    def test_train_state_changed(self):
        self.assertNotEqual(self.initial_state, self.after_train_state)

    def test_restored_state_same(self):
        self.assertDictEqual(self.initial_state, self.restored_state)
