from medcat.utils.helpers import has_spacy_model, ensure_spacy_model
from medcat.pipe import DEFAULT_SPACY_MODEL

import unittest
import subprocess


class HasSpacyModelTests(unittest.TestCase):

    def test_no_rubbish_model(self, model_name='rubbish_model'):
        self.assertFalse(has_spacy_model(model_name))

    def test_has_def_model(self, model_name=DEFAULT_SPACY_MODEL):
        self.assertTrue(has_spacy_model(model_name))


class EnsureSpacyModelTests(unittest.TestCase):

    def test_fails_rubbish_model(self, model_name='rubbish_model'):
        with self.assertRaises(subprocess.CalledProcessError):
            ensure_spacy_model(model_name)

    def test_success_def_model(self, model_name=DEFAULT_SPACY_MODEL):
        ensure_spacy_model(model_name)
