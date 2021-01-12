from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.ner.vocab_based_ner import NER
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.pipe import Pipe
from medcat.utils.normalizers import BasicSpellChecker
from medcat.vocab import Vocab
from medcat.preprocessing.cleaners import prepare_name
from medcat.linking.vector_context_model import ContextModel
from functools import partial
from medcat.linking.context_based_linker import Linker
from medcat.config import Config
import logging
from medcat.cdb import CDB
import os
import requests
import unittest


class A_NERTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #REVIEW - IS THIS THE SIMPLEST SETUP TO TEST NER
        print("Set up CDB")
        cls.config = Config()
        cls.config.general['log_level'] = logging.INFO
        cls.cdb = CDB(config=cls.config)

        print("Set up Vocab")
        vocab_path = "./tmp_vocab.dat"
        if not os.path.exists(vocab_path):
            tmp = requests.get("https://s3-eu-west-1.amazonaws.com/zkcl/vocab.dat")
            with open(vocab_path, 'wb') as f:
                f.write(tmp.content)

        cls.vocab = Vocab.load(vocab_path)


        print("Set up NLP pipeline")
        cls.nlp = Pipe(tokenizer=spacy_split_all, config=cls.config)
        cls.nlp.add_tagger(tagger=partial(tag_skip_and_punct, config=cls.config),
                       name='skip_and_punct',
                       additional_fields=['is_punct'])
        #CLARIFY WHY CDB.VOCAB HERE 
        cls.spell_checker = BasicSpellChecker(cdb_vocab=cls.cdb.vocab, config=cls.config, data_vocab=cls.vocab)
        cls.nlp.add_token_normalizer(spell_checker=cls.spell_checker, config=cls.config)
        cls.ner = NER(cls.cdb, cls.config)
        cls.nlp.add_ner(cls.ner)

        print("Set up Linker")
        cls.link = Linker(cls.cdb, cls.vocab, cls.config)
        cls.nlp.add_linker(cls.link)

        print("Set limits for tokens and uppercase")
        cls.config.ner['max_skip_tokens'] = 1
        cls.config.ner['upper_case_limit_len'] = 4
        cls.config.linking['disamb_length_limit'] = 2

        print("Add concepts")
        cls.cdb.add_names(cui='S-229004', names=prepare_name('Movar', cls.nlp, {}, cls.config))
        cls.cdb.add_names(cui='S-229004', names=prepare_name('Movar viruses', cls.nlp, {}, cls.config))
        cls.cdb.add_names(cui='S-229005', names=prepare_name('CDB', cls.nlp, {}, cls.config))

    def test_aa_cdb_names_output(self):
        target_result = {'S-229004': {'movar~virus', 'movar', 'movar~viruses'}, 'S-229005': {'cdb'}}
        self.assertEqual(self.cdb.cui2names, target_result)

    def test_ab_number_of_entities(self):
        self.text = "CDB - I was running and then Movar    Virus attacked and CDb"
        self.text_post_pipe = self.nlp(self.text)
        self.assertEqual(len(self.text_post_pipe._.ents), 2, "Should equal 2")

if __name__ == '__main__':
    unittest.main()
