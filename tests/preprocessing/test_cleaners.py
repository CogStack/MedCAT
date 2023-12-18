from medcat.preprocessing.cleaners import prepare_name
from medcat.config import Config
from medcat.cdb_maker import CDBMaker

import logging, os

import unittest


class BaseCDBMakerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config = Config()
        config.general['log_level'] = logging.DEBUG
        config.general["spacy_model"] = "en_core_web_md"
        cls.maker = CDBMaker(config)
        csvs = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'examples', 'cdb.csv'),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'examples', 'cdb_2.csv')
        ]
        cls.cdb = cls.maker.prepare_csvs(csvs, full_build=True)


class BasePrepareNameTest(BaseCDBMakerTests):
    raw_name = 'raw'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.do_prepare_name()

    # method called after setup, when raw_name has been specified
    @classmethod
    def do_prepare_name(cls) -> None:
        cls.name = cls.cdb.config.general.separator.join(cls.raw_name.split())
        cls.names = prepare_name(cls.raw_name, cls.maker.pipe.spacy_nlp, {}, cls.cdb.config)

    def _dict_has_key_val_type(self, d: dict, key, val_type):
        self.assertIn(key, d)
        self.assertIsInstance(d[key], val_type)

    def _names_has_key_val_type(self, key, val_type):
        self._dict_has_key_val_type(self.names, key, val_type)

    def test_result_has_name(self):
        self._names_has_key_val_type(self.name, dict)

    def test_name_info_has_tokens(self):
        self._dict_has_key_val_type(self.names[self.name], 'tokens', list)

    def test_name_info_has_words_as_tokens(self):
        name_info = self.names[self.name]
        tokens = name_info['tokens']
        for word in self.raw_name.split():
            with self.subTest(word):
                self.assertIn(word, tokens)
    

class NamePreparationTests_OneLetter(BasePrepareNameTest):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.raw_name = "a"
        # the minimum name length is defined by the following config option
        # if I don't set this to 1 here, I would see the tests fail
        # that would be because the result from `prepare_names` would be empty
        cls.cdb.config.cdb_maker.min_letters_required = 1
        super().do_prepare_name()
    

class NamePreparationTests_TwoLetters(BasePrepareNameTest):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.raw_name = "an"
        super().do_prepare_name()
    

class NamePreparationTests_MultiToken(BasePrepareNameTest):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.raw_name = "this raw name"
        super().do_prepare_name()
    

class NamePreparationTests_Empty(BaseCDBMakerTests):
    """In case of an empty name, I would expect the names dict
    returned by `prepare_name` to be empty.
    """
    empty_raw_name = ''

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.names = prepare_name(cls.empty_raw_name, cls.maker.pipe.spacy_nlp, {}, cls.cdb.config)

    def test_names_dict_is_empty(self):
        self.assertEqual(len(self.names), 0)
        self.assertEqual(self.names, {})
