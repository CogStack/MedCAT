import unittest
import logging
import os
import requests
from spacy.language import Language
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.config import Config
from medcat.pipe import Pipe
from medcat.meta_cat import MetaCAT
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.utils.normalizers import BasicSpellChecker
from medcat.ner.vocab_based_ner import NER
from medcat.linking.context_based_linker import Linker


class PipeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config()
        cls.config.general['log_level'] = logging.INFO
        cls.cdb = CDB(config=cls.config)

        vocab_path = "./tmp_vocab.dat"
        if not os.path.exists(vocab_path):
            tmp = requests.get("https://medcat.rosalind.kcl.ac.uk/media/vocab.dat")
            with open(vocab_path, 'wb') as f:
                f.write(tmp.content)

        cls.vocab = Vocab.load(vocab_path)
        cls.spell_checker = BasicSpellChecker(cdb_vocab=cls.cdb.vocab, config=cls.config, data_vocab=cls.vocab)
        cls.ner = NER(cls.cdb, cls.config)
        cls.linker = Linker(cls.cdb, cls.vocab, cls.config)
        cls.config.ner['max_skip_tokens'] = 1
        cls.config.ner['upper_case_limit_len'] = 4
        cls.config.linking['disamb_length_limit'] = 2
        cls.meta_cat = MetaCAT()
        cls.text = "CDB - I was running and then Movar Virus attacked and CDb"
        cls.config = Config()
        cls.config.general['log_level'] = logging.INFO
        cls.undertest = Pipe(tokenizer=spacy_split_all, config=cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.undertest.destroy_nlp()

    def setUp(self) -> None:
        PipeTests.undertest.force_remove(tag_skip_and_punct.__name__)
        PipeTests.undertest.force_remove("token_normalizer")
        PipeTests.undertest.force_remove(PipeTests.ner.name)
        PipeTests.undertest.force_remove(PipeTests.linker.name)
        PipeTests.undertest.force_remove(PipeTests.meta_cat.name)

    def test_add_tagger(self):
        PipeTests.undertest.add_tagger(tagger=tag_skip_and_punct, name="tag_skip_and_punct", additional_fields=["is_punct"])

        self.assertEqual(tag_skip_and_punct.__name__, Language.get_factory_meta(tag_skip_and_punct.__name__).factory)
        self.assertEqual(PipeTests.config, Language.get_factory_meta(tag_skip_and_punct.__name__).default_config["config"])

    def test_add_token_normalizer(self):
        PipeTests.undertest.add_token_normalizer(PipeTests.config, spell_checker=PipeTests.spell_checker)

        self.assertEqual("token_normalizer", Language.get_factory_meta("token_normalizer").factory)

    def test_add_ner(self):
        PipeTests.undertest.add_ner(PipeTests.ner)

        self.assertEqual(PipeTests.ner.name, Language.get_factory_meta(PipeTests.ner.name).factory)

    def test_add_linker(self):
        PipeTests.undertest.add_linker(PipeTests.linker)

        self.assertEqual(PipeTests.linker.name, Language.get_factory_meta(PipeTests.linker.name).factory)

    def test_add_meta_cat(self):
        PipeTests.undertest.add_meta_cat(PipeTests.meta_cat, "cat_name")

        self.assertEqual(PipeTests.meta_cat.name, Language.get_factory_meta(PipeTests.meta_cat.name).factory)
