import logging
import unittest
from spacy.lang.en import English
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.ner.vocab_based_ner import NER
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.pipe import Pipe
from medcat.utils.normalizers import BasicSpellChecker
from medcat.vocab import Vocab
from medcat.preprocessing.cleaners import prepare_name
from medcat.linking.context_based_linker import Linker
from medcat.config import Config
from medcat.cdb import CDB

from .helper import VocabDownloader


class A_NERTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Set up CDB")
        cls.config = Config()
        cls.config.general['log_level'] = logging.INFO
        cls.config.general["spacy_model"] = "en_core_web_md"
        cls.cdb = CDB(config=cls.config)

        print("Set up Vocab")
        downloader = VocabDownloader()
        vocab_path = downloader.vocab_path
        downloader.check_or_download()

        cls.vocab = Vocab.load(vocab_path)


        print("Set up NLP pipeline")
        cls.pipe = Pipe(tokenizer=spacy_split_all, config=cls.config)
        cls.pipe.add_tagger(tagger=tag_skip_and_punct,
                            name='skip_and_punct',
                            additional_fields=['is_punct'])

        cls.spell_checker = BasicSpellChecker(cdb_vocab=cls.cdb.vocab, config=cls.config, data_vocab=cls.vocab)
        cls.pipe.add_token_normalizer(spell_checker=cls.spell_checker, config=cls.config)
        cls.ner = NER(cls.cdb, cls.config)
        cls.pipe.add_ner(cls.ner)

        print("Set up Linker")
        cls.link = Linker(cls.cdb, cls.vocab, cls.config)
        cls.pipe.add_linker(cls.link)

        print("Set limits for tokens and uppercase")
        cls.config.ner['max_skip_tokens'] = 1
        cls.config.ner['upper_case_limit_len'] = 4
        cls.config.linking['disamb_length_limit'] = 2
        cls.config.general["spacy_model"] = "en_core_sci_sm"

        print("Add concepts")
        cls.cdb.add_names(cui='S-229004', names=prepare_name('Movar', cls.pipe, {}, cls.config))
        cls.cdb.add_names(cui='S-229004', names=prepare_name('Movar viruses', cls.pipe, {}, cls.config))
        cls.cdb.add_names(cui='S-229005', names=prepare_name('CDB', cls.pipe, {}, cls.config))

        print("Add test text")
        cls.text = "CDB - I was running and then Movar    Viruses attacked and CDb"
        cls.text_post_pipe = cls.pipe(cls.text)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.pipe.destroy()

    def test_aa_cdb_names_output(self):
        print("Fixing 'movar~viruse' -> 'movar-virus' for newere en_core_web_md")
        target_result = {'S-229004': {'movar~virus', 'movar', 'movar~viruses'}, 'S-229005': {'cdb'}}
        self.assertEqual(self.cdb.cui2names, target_result)

    def test_ab_entities_length(self):
        self.assertEqual(len(self.text_post_pipe._.ents), 2, "Should equal 2")

    def test_ac_entities_linked_candidates(self):
        target_result = 'S-229004'
        self.assertEqual(self.text_post_pipe._.ents[0]._.link_candidates[0], target_result)

    def test_ad_max_skip_entities_length(self):
        self.config.ner['max_skip_tokens'] = 3
        self.text_post_pipe = self.pipe(self.text)
        self.assertEqual(len(self.text_post_pipe._.ents), 3, "Should equal 3")

    def test_ae_upper_case_entities_length(self):
        self.config.ner['upper_case_limit_len'] = 3
        self.text_post_pipe = self.pipe(self.text)
        self.assertEqual(len(self.text_post_pipe._.ents), 4, "Should equal 4")

    def test_af_min_name_entities_length(self):
        self.config.ner['min_name_len'] = 4
        self.text_post_pipe = self.pipe(self.text)
        print(self.text)
        print(self.text_post_pipe._.ents)
        self.assertEqual(len(self.text_post_pipe._.ents), 2, "Should equal 2")


if __name__ == '__main__':
    unittest.main()
