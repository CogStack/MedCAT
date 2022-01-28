import os
import unittest
import tempfile
import asyncio
import json
from unittest.mock import patch
from tests.helper import AsyncMock
from medcat.utils.checkpoint import Checkpoint


class CheckpointTest(unittest.TestCase):

    def test_restore(self):
        dir_path = os.path.join(os.path.dirname(__file__), "..", "resources", "checkpoints", "cat_train")

        checkpoint = Checkpoint.restore(dir_path)

        self.assertEqual(2, checkpoint.steps)
        self.assertEqual(20, checkpoint.count)
        self.assertEqual(1, checkpoint.max_to_keep)

    def test_purge(self):
        dir_path = tempfile.TemporaryDirectory()
        ckpt_file_path = os.path.join(dir_path.name, "checkpoint-1-1")
        open(os.path.join(dir_path.name, "checkpoint-1-1"), "w").close()
        checkpoint = Checkpoint(dir_path=dir_path.name, steps=1, max_to_keep=1)
        self.assertTrue(os.path.isfile(ckpt_file_path))

        checkpoint.purge()

        checkpoints = [f for f in os.listdir(dir_path.name) if "checkpoint-" in f]
        self.assertEqual([], checkpoints)

    @patch("medcat.cdb.CDB")
    def test_populate(self, cdb):
        dir_path = os.path.join(os.path.dirname(__file__), "..", "resources", "checkpoints", "cat_train")
        checkpoint = Checkpoint.restore(dir_path)

        checkpoint.populate(cdb)

        cdb.load.assert_called_with(os.path.abspath(os.path.join(dir_path, "checkpoint-2-20")))

    @patch("medcat.cdb.CDB")
    def test_save(self, cdb):
        dir_path = tempfile.TemporaryDirectory()
        checkpoint = Checkpoint(dir_path=dir_path.name, steps=1, max_to_keep=1)

        checkpoint.save(cdb, 1)

        cdb.save.assert_called()
        self.assertEqual(1, checkpoint.steps)
        self.assertEqual(1, checkpoint.max_to_keep)
        self.assertEqual(1, checkpoint.count)

    @patch("medcat.cdb.CDB", new_callable=AsyncMock)
    def test_save_async(self, cdb):
        dir_path = tempfile.TemporaryDirectory()
        checkpoint = Checkpoint(dir_path=dir_path.name, steps=1, max_to_keep=1)

        asyncio.run(checkpoint.save_async(cdb, 1))

        cdb.save_async.assert_called()
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
        self.assertEqual("Argument at position 1 is not a positive integer", str(e2.exception))

    def test_validation_on_max_to_keep(self):
        with self.assertRaises(Exception) as e1:
            Checkpoint(dir_path="dir_path", steps=1000, max_to_keep=-1)
        self.assertEqual("Argument 'max_to_keep' is not a positive integer", str(e1.exception))

        with self.assertRaises(Exception) as e2:
            checkpoint = Checkpoint(dir_path="dir_path", steps=1000, max_to_keep=1)
            checkpoint.max_to_keep = -1
        self.assertEqual("Argument at position 1 is not a positive integer", str(e2.exception))
