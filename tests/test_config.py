import unittest
import pickle
import tempfile
from medcat.config import Config


class ConfigTests(unittest.TestCase):

    def test_pickleability(self):
        with tempfile.TemporaryFile() as f:
            pickle.dump(Config(), f)

    def test_from_dict(self):
        config = Config.from_dict({"key": "value"})
        self.assertEqual("value", config.key)


if __name__ == '__main__':
    unittest.main()
