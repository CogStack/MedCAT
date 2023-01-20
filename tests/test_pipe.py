import unittest
import logging
from spacy.language import Language
from medcat.cdb import CDB
from medcat.vocab import Vocab
from medcat.config import Config
from medcat.pipe import Pipe
from medcat.meta_cat import MetaCAT
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.utils.normalizers import BasicSpellChecker, TokenNormalizer
from medcat.ner.vocab_based_ner import NER
from medcat.linking.context_based_linker import Linker
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBERT
from transformers import AutoTokenizer


from .helper import VocabDownloader


class PipeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config()
        cls.config.general['log_level'] = logging.INFO
        cls.config.general["spacy_model"] = "en_core_web_md"
        cls.config.ner['max_skip_tokens'] = 1
        cls.config.ner['upper_case_limit_len'] = 4
        cls.config.linking['disamb_length_limit'] = 2
        cls.cdb = CDB(config=cls.config)

        downloader = VocabDownloader()
        vocab_path = downloader.vocab_path
        downloader.check_or_download()

        cls.vocab = Vocab.load(vocab_path)
        cls.spell_checker = BasicSpellChecker(cdb_vocab=cls.cdb.vocab, config=cls.config, data_vocab=cls.vocab)
        cls.ner = NER(cls.cdb, cls.config)
        cls.linker = Linker(cls.cdb, cls.vocab, cls.config)

        _tokenizer = TokenizerWrapperBERT(hf_tokenizers=AutoTokenizer.from_pretrained("bert-base-uncased"))
        cls.meta_cat = MetaCAT(tokenizer=_tokenizer)

        cls.text = "CDB - I was running and then Movar Virus attacked and CDb"
        cls.undertest = Pipe(tokenizer=spacy_split_all, config=cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.undertest.destroy()

    def setUp(self) -> None:
        PipeTests.undertest.force_remove(tag_skip_and_punct.name)
        PipeTests.undertest.force_remove(TokenNormalizer.name)
        PipeTests.undertest.force_remove(PipeTests.ner.name)
        PipeTests.undertest.force_remove(PipeTests.linker.name)
        PipeTests.undertest.force_remove(PipeTests.meta_cat.name)

    def test_add_tagger(self):
        PipeTests.undertest.add_tagger(tagger=tag_skip_and_punct, name=tag_skip_and_punct.name, additional_fields=["is_punct"])

        self.assertEqual(tag_skip_and_punct.name, Language.get_factory_meta(tag_skip_and_punct.name).factory)
        self.assertEqual(PipeTests.config, Language.get_factory_meta(tag_skip_and_punct.name).default_config["config"])

    def test_add_token_normalizer(self):
        PipeTests.undertest.add_token_normalizer(PipeTests.config, spell_checker=PipeTests.spell_checker)

        self.assertEqual(TokenNormalizer.name, Language.get_factory_meta(TokenNormalizer.name).factory)

    def test_add_ner(self):
        PipeTests.undertest.add_ner(PipeTests.ner)

        self.assertEqual(PipeTests.ner.name, Language.get_factory_meta(PipeTests.ner.name).factory)

    def test_add_linker(self):
        PipeTests.undertest.add_linker(PipeTests.linker)

        self.assertEqual(PipeTests.linker.name, Language.get_factory_meta(PipeTests.linker.name).factory)

    def test_add_meta_cat(self):
        PipeTests.undertest.add_meta_cat(PipeTests.meta_cat)

        self.assertEqual(PipeTests.meta_cat.name, Language.get_factory_meta(PipeTests.meta_cat.name).factory)

    def test_batch_multi_process(self):
        PipeTests.undertest.add_tagger(tagger=tag_skip_and_punct, additional_fields=["is_punct"])
        PipeTests.undertest.add_token_normalizer(PipeTests.config, spell_checker=PipeTests.spell_checker)
        PipeTests.undertest.add_ner(PipeTests.ner)
        PipeTests.undertest.add_linker(PipeTests.linker)
        PipeTests.undertest.add_meta_cat(PipeTests.meta_cat)

        PipeTests.undertest.set_error_handler(_error_handler)
        docs = list(self.undertest.batch_multi_process([PipeTests.text, PipeTests.text, PipeTests.text], n_process=1, batch_size=1))
        PipeTests.undertest.reset_error_handler()

        self.assertEqual(3, len(docs))
        self.assertEqual(PipeTests.text, docs[0].text)
        self.assertEqual(PipeTests.text, docs[1].text)
        self.assertEqual(PipeTests.text, docs[2].text)

    def test_callable_with_generated_texts(self):
        def _generate_texts(texts):
            yield from texts

        PipeTests.undertest.add_tagger(tagger=tag_skip_and_punct, additional_fields=["is_punct"])
        PipeTests.undertest.add_token_normalizer(PipeTests.config, spell_checker=PipeTests.spell_checker)
        PipeTests.undertest.add_ner(PipeTests.ner)
        PipeTests.undertest.add_linker(PipeTests.linker)
        PipeTests.undertest.add_meta_cat(PipeTests.meta_cat)

        docs = list(self.undertest(_generate_texts([PipeTests.text, None, PipeTests.text])))

        self.assertEqual(3, len(docs))
        self.assertEqual(PipeTests.text, docs[0].text)
        self.assertIsNone(docs[1])
        self.assertEqual(PipeTests.text, docs[2].text)

    def test_callable_with_single_text(self):
        PipeTests.undertest.add_tagger(tagger=tag_skip_and_punct, additional_fields=["is_punct"])
        PipeTests.undertest.add_token_normalizer(PipeTests.config, spell_checker=PipeTests.spell_checker)
        PipeTests.undertest.add_ner(PipeTests.ner)
        PipeTests.undertest.add_linker(PipeTests.linker)
        PipeTests.undertest.add_meta_cat(PipeTests.meta_cat)

        doc = self.undertest(PipeTests.text)

        self.assertEqual(PipeTests.text, doc.text)

    def test_callable_with_multi_texts(self):
        PipeTests.undertest.add_tagger(tagger=tag_skip_and_punct, additional_fields=["is_punct"])
        PipeTests.undertest.add_token_normalizer(PipeTests.config, spell_checker=PipeTests.spell_checker)
        PipeTests.undertest.add_ner(PipeTests.ner)
        PipeTests.undertest.add_linker(PipeTests.linker)
        PipeTests.undertest.add_meta_cat(PipeTests.meta_cat)

        docs = list(self.undertest([PipeTests.text, None, PipeTests.text]))

        self.assertEqual(3, len(docs))
        self.assertEqual(PipeTests.text, docs[0].text)
        self.assertIsNone(docs[1])
        self.assertEqual(PipeTests.text, docs[2].text)

def _error_handler(proc_name, proc, docs, e):
    print("Exception raised when when applying component {}".format(proc_name))


if __name__ == '__main__':
    unittest.main()
