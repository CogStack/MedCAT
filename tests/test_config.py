import unittest
import pickle
import tempfile
from medcat.config import Config


class ConfigTests(unittest.TestCase):

    def test_identifier_works(self):
        c = Config()
        self.assertIsNotNone(c.general.spacy_model)

    def test_legacy_getitem_works(self):
        c = Config()
        self.assertIsNotNone(c.general['spacy_model'])

    def test_creates_new_defaults(self):
        # the default values are mutable
        # so if new instances weren't being created,
        # the second one would refer to the same instance
        # for the .general field as the first and as such
        # this test would fail
        bc1 = Config()
        bc1.general.spacy_model = 'smth'
        bc2 = Config()
        print(bc2.general.spacy_model, 'vs', bc1.general.spacy_model)
        self.assertNotEqual(bc2.general.spacy_model, bc1.general.spacy_model)

    def test_uses_different_internal_instances(self):
        # again, if the instance for .general was the same,
        # we would see this test fail
        bc1: Config = Config()
        bc2: Config = Config()
        self.assertEqual(bc1, bc2)
        bc1.general.spacy_model = 'Non-default-value'
        self.assertNotEqual(bc1.general, bc2.general)
        self.assertNotEqual(bc1.general.spacy_model, bc2.general.spacy_model)
        self.assertNotEqual(bc1, bc2)

    def test_pickleability(self):
        with tempfile.TemporaryFile() as f:
            pickle.dump(Config(), f)

    def test_from_dict(self):
        config = Config.from_dict({"key": "value"})
        self.assertEqual("value", config.key)


if __name__ == '__main__':
    unittest.main()
