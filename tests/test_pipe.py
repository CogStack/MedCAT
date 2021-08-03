import unittest
import logging
import os
import requests
from spacy.language import Language
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.config import Config
from medcat.pipe import Pipe
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

        print("Set up Vocab")
        vocab_path = "./tmp_vocab.dat"
        if not os.path.exists(vocab_path):
            tmp = requests.get("https://medcat.rosalind.kcl.ac.uk/media/vocab.dat")
            with open(vocab_path, 'wb') as f:
                f.write(tmp.content)

        cls.vocab = Vocab.load(vocab_path)
        cls.spell_checker = BasicSpellChecker(cdb_vocab=cls.cdb.vocab, config=cls.config, data_vocab=cls.vocab)
        cls.ner = NER(cls.cdb, cls.config)
        cls.link = Linker(cls.cdb, cls.vocab, cls.config)
        cls.config.ner['max_skip_tokens'] = 1
        cls.config.ner['upper_case_limit_len'] = 4
        cls.config.linking['disamb_length_limit'] = 2
        cls.text = "CDB - I was running and then Movar Virus attacked and CDb"

    def setUp(self) -> None:
        self.config = Config()
        self.config.general['log_level'] = logging.INFO
        self.undertest = Pipe(tokenizer=spacy_split_all, config=self.config)

    def test_add_tagger(self):
        self.assertRaises(ValueError, lambda: Language.get_factory_meta("tag_skip_and_punct"))

        self.undertest.add_tagger(tagger=tag_skip_and_punct, name="skip_and_punct", additional_fields=["is_punct"])

        self.assertEqual("tag_skip_and_punct", Language.get_factory_meta("tag_skip_and_punct").factory)
        self.assertEqual(self.config, Language.get_factory_meta("tag_skip_and_punct").default_config["config"])

    def test_add_token_normalizer(self):
        self.assertRaises(ValueError, lambda: Language.get_factory_meta("TokenNormalizer"))

        self.undertest.add_token_normalizer(spell_checker=PipeTests.spell_checker, config=PipeTests.config)

        self.assertEqual("TokenNormalizer", Language.get_factory_meta("TokenNormalizer").factory)

    def test_batch_process(self):
        docs = list(self.undertest.batch_process([PipeTests.text, PipeTests.text]))
        self.assertEqual(2, len(docs))
        self.assertEqual(PipeTests.text, docs[0].text)
        self.assertEqual(PipeTests.text, docs[1].text)
