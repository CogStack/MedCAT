import unittest
import pickle
import tempfile
from medcat.config import Config, MixingConfig, VersionInfo, General
from pydantic import ValidationError
import os


class ConfigTests(unittest.TestCase):
    _NON_DEFAULT_NON_NONE_FIELDS = ['word_skipper', 'punct_checker']
    # The values for these config options are not default
    # since they are recalculated at init time
    # however, they are also not None

    def test_identifier_works(self):
        c = Config()
        self.assertIsNotNone(c.general.spacy_model)

    def test_identifier_works_all(self):
        # Ignoring work_skipper and punct checker since their values
        # are not default, but they are also not None
        def check_recursively(c: MixingConfig):
            for field_name, field in c.fields().items():
                with self.subTest(f'{type(c).__name__}:{field_name}'):
                    val = getattr(c, field_name)
                    if field_name in ConfigTests._NON_DEFAULT_NON_NONE_FIELDS:
                        self.assertIsNotNone(val)
                    else:
                        self.assertEqual(val, field.default)
                if isinstance(val, MixingConfig):
                    check_recursively(val)
        check_recursively(Config())

    def test_legacy_getitem_works(self):
        c = Config()
        self.assertIsNotNone(c.general['spacy_model'])

    def test_getitem_works_all(self):
        # Ignoring work_skipper and punct checker since their values
        # are not default, but they are also not None
        def check_recursively(c: MixingConfig):
            for field_name, field in c.fields().items():
                with self.subTest():
                    val = c[field_name]
                    if field_name in ConfigTests._NON_DEFAULT_NON_NONE_FIELDS:
                        self.assertIsNotNone(val)
                    else:
                        self.assertEqual(val, field.default)
                if isinstance(val, MixingConfig):
                    check_recursively(val)
        check_recursively(Config())

    def test_creates_new_defaults(self):
        # the default values are mutable
        # so if new instances weren't being created,
        # the second one would refer to the same instance
        # for the .general field as the first and as such
        # this test would fail
        bc1 = Config()
        bc1.general.spacy_model = 'smth'
        bc2 = Config()
        self.assertNotEqual(bc2.general.spacy_model, bc1.general.spacy_model)

    def test_config_get_hash_gets_same_value_on_each_pass(self):
        c = Config()
        h1 = c.get_hash()
        h2 = c.get_hash()
        self.assertEqual(h1, h2)

    def test_identical_config_has_same_hash(self):
        c1 = Config()
        c2 = Config()
        self.assertEqual(c1.get_hash(), c2.get_hash())

    def test_ignored_parts_do_not_affect_hash(self, ignored_constructors={'version': lambda: VersionInfo(history=['-1.42.0'])}):
        c = Config()
        orig_hash = c.get_hash()
        for name, constr in ignored_constructors.items():
            with self.subTest(f'Changing {name}'):
                new_version = constr()
                setattr(c, name, new_version)
                hash = c.get_hash()
                self.assertEqual(orig_hash, hash)

    def test_changed_config_different_hash(self):
        c = Config()
        hash1 = c.get_hash()
        c.general.log_format = ''
        hash2 = c.get_hash()
        self.assertNotEqual(hash1, hash2)

    def test_config_get_hash_gets_same_value_after_change_back_main(self):
        c = Config()
        h1 = c.get_hash()
        # save prev
        prev_gen = c.general
        # change
        gen = General(log_format='LOG')
        c.general = gen
        h2 = c.get_hash()
        self.assertNotEqual(h1, h2)
        # set back
        c.general = prev_gen
        h3 = c.get_hash()
        self.assertEqual(h1, h3)

    def test_config_get_hash_gets_same_value_after_change_back_nested(self):
        c = Config()
        h1 = c.get_hash()
        # save prev
        prev_log_format = str(c.general.log_format)
        # change
        c.general.log_format = ''
        h2 = c.get_hash()
        self.assertNotEqual(h1, h2)
        # set back
        c.general.log_format = prev_log_format
        h3 = c.get_hash()
        self.assertEqual(h1, h3)

    def test_lists_are_different(self):
        bc1 = Config()
        bc1.version.history.append('v3')
        bc2 = Config()
        self.assertNotEqual(bc1.version.history, bc2.version.history)

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

    def test_fails_upon_wrong_type_init(self):
        with self.assertRaises(ValidationError):
            VersionInfo(history=-1)

    def test_fails_upon_wrong_type_merge(self):
        with self.assertRaises(ValidationError):
            VersionInfo.from_dict(dict(history=-1))

    def test_fails_upon_wrong_type_assignment(self):
        vi = VersionInfo()
        with self.assertRaises(ValidationError):
            vi.history = -1

    def test_parsing(self, folder=os.path.join('tests', 'model_creator'),
                     files=('medcat.txt',),  # file(s) to read
                     # the list of lambdas for attributes that differ, for each file (if needed)
                     differences=([
                         lambda c: c.preprocessing.do_not_normalize, lambda c: c.general.diacritics,
                         lambda c: c.ner.check_upper_case_names, lambda c: c.version.location,
                         lambda c: c.version.ontology, lambda c: c.version.location],)
                     ):
        for file, getters in zip(files, differences):
            with self.subTest(f'Checking {file}'):
                full_file = os.path.join(folder, file)
                c = Config()
                h1 = c.get_hash()
                c.parse_config_file(full_file)
                self.assertIsInstance(c, Config)
                h2 = c.get_hash()
                self.assertNotEqual(h1, h2)
                _def_config = Config()
                for i, val_getter in enumerate(getters):
                    with self.subTest(f'Checking getter {val_getter} (#{i}) for {file}'):
                        v1, v2 = val_getter(_def_config), val_getter(c)
                        self.assertNotEqual(v1, v2)

    def test_pickleability(self):
        with tempfile.TemporaryFile() as f:
            pickle.dump(Config(), f)

    def test_from_dict(self):
        config = Config.from_dict({"key": "value"})
        self.assertEqual("value", config.key)


if __name__ == '__main__':
    unittest.main()
