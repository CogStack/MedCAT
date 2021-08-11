import unittest
import pickle
import tempfile
from medcat.config import Config


class ConfigTests(unittest.TestCase):

    def test_pickleability(self):
        with tempfile.TemporaryFile() as f:
            pickle.dump(Config(), f)
