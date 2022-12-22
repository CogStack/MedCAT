import logging
import unittest
import numpy as np
from timeit import default_timer as timer
from medcat.cdb import CDB
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.ner.vocab_based_ner import NER
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.pipe import Pipe
from medcat.utils.normalizers import BasicSpellChecker
from medcat.vocab import Vocab
from medcat.preprocessing.cleaners import prepare_name
from medcat.linking.vector_context_model import ContextModel
from medcat.linking.context_based_linker import Linker
from medcat.config import Config

from ..helper import VocabDownloader


class NerArchiveTests(unittest.TestCase):

    def setUp(self) -> None:
        self.config = Config()
        self.config.general['log_level'] = logging.INFO
        cdb = CDB(config=self.config)

        self.nlp = Pipe(tokenizer=spacy_split_all, config=self.config)
        self.nlp.add_tagger(tagger=tag_skip_and_punct,
                       name='skip_and_punct',
                       additional_fields=['is_punct'])

        # Add a couple of names
        cdb.add_names(cui='S-229004', names=prepare_name('Movar', self.nlp, {}, self.config))
        cdb.add_names(cui='S-229004', names=prepare_name('Movar viruses', self.nlp, {}, self.config))
        cdb.add_names(cui='S-229005', names=prepare_name('CDB', self.nlp, {}, self.config))
        # Check
        #assert cdb.cui2names == {'S-229004': {'movar', 'movarvirus', 'movarviruses'}, 'S-229005': {'cdb'}}

        downloader = VocabDownloader()
        self.vocab_path = downloader.vocab_path
        downloader.check_or_download()

        vocab = Vocab.load(self.vocab_path)
        # Make the pipeline
        self.nlp = Pipe(tokenizer=spacy_split_all, config=self.config)
        self.nlp.add_tagger(tagger=tag_skip_and_punct,
                       name='skip_and_punct',
                       additional_fields=['is_punct'])
        spell_checker = BasicSpellChecker(cdb_vocab=cdb.vocab, config=self.config, data_vocab=vocab)
        self.nlp.add_token_normalizer(spell_checker=spell_checker, config=self.config)
        ner = NER(cdb, self.config)
        self.nlp.add_ner(ner)

        # Add Linker
        link = Linker(cdb, vocab, self.config)
        self.nlp.add_linker(link)

        self.text = "CDB - I was running and then Movar    Virus attacked and CDb"

    def tearDown(self) -> None:
        self.nlp.destroy()

    def test_limits_for_tokens_and_uppercase(self):
        self.config.ner['max_skip_tokens'] = 1
        self.config.ner['upper_case_limit_len'] = 4
        self.config.linking['disamb_length_limit'] = 2

        d = self.nlp(self.text)

        assert len(d._.ents) == 2
        assert d._.ents[0]._.link_candidates[0] == 'S-229004'

    def test_change_limit_for_skip(self):
        self.config.ner['max_skip_tokens'] = 3
        d = self.nlp(self.text)
        assert len(d._.ents) == 3

    def test_change_limit_for_upper_case(self):
        self.config.ner['upper_case_limit_len'] = 3
        d = self.nlp(self.text)
        assert len(d._.ents) == 4

    def test_check_name_length_limit(self):
        self.config.ner['min_name_len'] = 4
        d = self.nlp(self.text)
        assert len(d._.ents) == 2

    def test_speed(self):
        text = "CDB - I was running and then Movar    Virus attacked and CDb"
        text = text * 300
        self.config.general['spell_check'] = True
        start = timer()
        for i in range(50):
            d = self.nlp(text)
        end = timer()
        print("Time: ", end - start)

    def test_without_spell_check(self):
        # Now without spell check
        self.config.general['spell_check'] = False
        start = timer()
        for i in range(50):
            d = self.nlp(self.text)
        end = timer()
        print("Time: ", end - start)


    def test_for_linker(self):
        self.config = Config()
        self.config.general['log_level'] = logging.DEBUG
        cdb = CDB(config=self.config)

        # Add a couple of names
        cdb.add_names(cui='S-229004', names=prepare_name('Movar', self.nlp, {}, self.config))
        cdb.add_names(cui='S-229004', names=prepare_name('Movar viruses', self.nlp, {}, self.config))
        cdb.add_names(cui='S-229005', names=prepare_name('CDB', self.nlp, {}, self.config))
        cdb.add_names(cui='S-2290045', names=prepare_name('Movar', self.nlp, {}, self.config))
        # Check
        #assert cdb.cui2names == {'S-229004': {'movar', 'movarvirus', 'movarviruses'}, 'S-229005': {'cdb'}, 'S-2290045': {'movar'}}

        cuis = list(cdb.cui2names.keys())
        for cui in cuis[0:50]:
            vectors = {'short': np.random.rand(300),
                      'long': np.random.rand(300),
                      'medium': np.random.rand(300)
                      }
            cdb.update_context_vector(cui, vectors, negative=False)

        d = self.nlp(self.text)
        vocab = Vocab.load(self.vocab_path)
        cm = ContextModel(cdb, vocab, self.config)
        cm.train_using_negative_sampling('S-229004')
        self.config.linking['train_count_threshold'] = 0

        cm.train('S-229004', d._.ents[1], d)

        cm.similarity('S-229004', d._.ents[1], d)

        cm.disambiguate(['S-2290045', 'S-229004'], d._.ents[1], 'movar', d)
