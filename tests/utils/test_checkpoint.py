import os
import unittest
import tempfile
import json
from unittest.mock import patch
from tests.helper import AsyncMock
from medcat.utils.checkpoint import Checkpoint, CheckpointConfig, CheckpointManager
from medcat.cdb import CDB


class CheckpointTest(unittest.TestCase):

    def test_from_latest(self):
        dir_path = os.path.join(os.path.dirname(__file__), "..", "resources", "checkpoints", "cat_train", "1643822916")

        checkpoint = Checkpoint.from_latest(dir_path)

        self.assertEqual(2, checkpoint.steps)
        self.assertEqual(20, checkpoint.count)
        self.assertEqual(1, checkpoint.max_to_keep)

    @patch("medcat.cdb.CDB.load", return_value="cdb_object")
    def test_restore_latest_cdb(self, cdb_load):
        dir_path = os.path.join(os.path.dirname(__file__), "..", "resources", "checkpoints", "cat_train", "1643822916")
        checkpoint = Checkpoint(dir_path=dir_path, steps=1, max_to_keep=5)

        cdb = checkpoint.restore_latest_cdb()

        self.assertEqual("cdb_object", cdb)
        self.assertEqual(1, checkpoint.steps)
        self.assertEqual(20, checkpoint.count)
        self.assertEqual(5, checkpoint.max_to_keep)
        cdb_load.assert_called_with(os.path.abspath(os.path.join(dir_path, "checkpoint-2-20")))

    @patch("medcat.cdb.CDB")
    def test_save(self, cdb):
        dir_path = tempfile.TemporaryDirectory()
        checkpoint = Checkpoint(dir_path=dir_path.name, steps=1, max_to_keep=1)

        checkpoint.save(cdb, 1)

        cdb.save.assert_called()
        self.assertEqual(1, checkpoint.steps)
        self.assertEqual(1, checkpoint.max_to_keep)
        self.assertEqual(1, checkpoint.count)

    def test_validation_on_steps(self):
        with self.assertRaises(Exception) as e1:
            Checkpoint(dir_path="dir_path", steps=0, max_to_keep=1)
        self.assertEqual("Argument 'steps' is not a positive integer", str(e1.exception))

        with self.assertRaises(Exception) as e2:
            checkpoint = Checkpoint(dir_path="dir_path", steps=1000, max_to_keep=1)
            checkpoint.steps = 0
        self.assertEqual("Argument at position 0 is not a positive integer", str(e2.exception))

    def test_validation_on_max_to_keep(self):
        with self.assertRaises(Exception) as e1:
            Checkpoint(dir_path="dir_path", steps=1000, max_to_keep=-1)
        self.assertEqual("Argument 'max_to_keep' is not a positive integer", str(e1.exception))

        with self.assertRaises(Exception) as e2:
            checkpoint = Checkpoint(dir_path="dir_path", steps=1000, max_to_keep=1)
            checkpoint.max_to_keep = -1
        self.assertEqual("Argument at position 0 is not a positive integer", str(e2.exception))

    def test_create_checkpoint(self):
        ckpt_out_dir_path = tempfile.TemporaryDirectory().name
        ckpt_config = {
            'output_dir': ckpt_out_dir_path,
            'steps': 1000,
            "max_to_keep": 5,
        }

        checkpoint = CheckpointManager("cat_train", CheckpointConfig(**ckpt_config)).create_checkpoint()

        self.assertTrue("cat_train" in checkpoint.dir_path)
        self.assertEqual(1000, checkpoint.steps)
        self.assertEqual(0, checkpoint.count)
        self.assertEqual(5, checkpoint.max_to_keep)

    def test_get_latest_checkpoint(self):
        ckpt_out_dir_path = os.path.join(os.path.dirname(__file__), "..", "resources", "checkpoints", "cat_train_supervised")
        ckpt_config = {
            'output_dir': ckpt_out_dir_path,
            'steps': 1000,
            "max_to_keep": 5,
        }

        checkpoint = CheckpointManager("cat_train_supervised", CheckpointConfig(**ckpt_config)).get_latest_checkpoint(ckpt_out_dir_path)

        self.assertTrue("1643823460" in checkpoint.dir_path)
        self.assertEqual(1, checkpoint.count)
        self.assertEqual(1000, checkpoint.steps)
        self.assertEqual(5, checkpoint.max_to_keep)

    def test_get_latest_training_dir(self):
        ckpt_out_dir_path = os.path.join(os.path.dirname(__file__), "..", "resources", "checkpoints", "cat_train")

        ckpt_dir_path = CheckpointManager.get_latest_training_dir(ckpt_out_dir_path)

        self.assertTrue("1643823460" in ckpt_dir_path)
