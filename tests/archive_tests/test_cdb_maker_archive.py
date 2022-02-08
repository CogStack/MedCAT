import logging
import unittest
import numpy as np
from medcat.cdb import CDB
from medcat.cdb_maker import CDBMaker
from medcat.config import Config
from medcat.preprocessing.cleaners import prepare_name


class CdbMakerArchiveTests(unittest.TestCase):

    def setUp(self):
        self.config = Config()
        self.config.general['log_level'] = logging.DEBUG
        self.maker = CDBMaker(self.config)

        # Building a new CDB from two files (full_build)
        csvs = ['../examples/cdb.csv', '../examples/cdb_2.csv']
        self.cdb = self.maker.prepare_csvs(csvs, full_build=True)

    def test_prepare_csvs(self):
        assert len(self.cdb.cui2names) == 3
        assert len(self.cdb.cui2snames) == 3
        assert len(self.cdb.name2cuis) == 5
        assert len(self.cdb.cui2tags) == 3
        assert len(self.cdb.cui2preferred_name) == 2
        assert len(self.cdb.cui2context_vectors) == 3
        assert len(self.cdb.cui2count_train) == 3
        assert self.cdb.name2cuis2status['virus']['C0000039'] == 'P'
        assert self.cdb.cui2type_ids['C0000039'] == {'T234', 'T109', 'T123'}
        assert self.cdb.addl_info['cui2original_names']['C0000039'] == {'Virus', 'Virus K', 'Virus M', 'Virus Z'}
        assert self.cdb.addl_info['cui2description']['C0000039'].startswith("Synthetic")

    def test_name_addition(self):
        self.cdb.add_names(cui='C0000239', names=prepare_name('MY: new,-_! Name.', self.maker.pipe.get_spacy_nlp(), {}, self.config), name_status='P', full_build=True)
        assert self.cdb.addl_info['cui2original_names']['C0000239'] == {'MY: new,-_! Name.', 'Second csv'}
        assert 'my:newname.' in self.cdb.name2cuis
        assert 'my:new' in self.cdb.snames
        assert 'my:newname.' in self.cdb.name2cuis2status
        assert self.cdb.name2cuis2status['my:newname.'] == {'C0000239': 'P'}

    def test_name_removal(self):
        self.cdb.remove_names(cui='C0000239', names=prepare_name('MY: new,-_! Name.', self.maker.pipe.get_spacy_nlp(), {}, self.config))
        # Run again to make sure it does not break anything
        self.cdb.remove_names(cui='C0000239', names=prepare_name('MY: new,-_! Name.', self.maker.pipe.get_spacy_nlp(), {}, self.config))
        assert len(self.cdb.name2cuis) == 5
        assert 'my:newname.' not in self.cdb.name2cuis2status

    def test_filtering(self):
        cuis_to_keep = {'C0000039'} # Because of transition 2 will be kept
        self.cdb.filter_by_cui(cuis_to_keep=cuis_to_keep)
        assert len(self.cdb.cui2names) == 2
        assert len(self.cdb.name2cuis) == 4
        assert len(self.cdb.snames) == 4

    def test_vector_addition(self):
        self.cdb.reset_training()
        np.random.seed(11)
        cuis = list(self.cdb.cui2names.keys())
        for i in range(2):
            for cui in cuis:
                vectors = {}
                for cntx_type in self.config.linking['context_vector_sizes']:
                    vectors[cntx_type] = np.random.rand(300)
                self.cdb.update_context_vector(cui, vectors, negative=False)

        assert self.cdb.cui2count_train['C0000139'] == 2
        assert self.cdb.cui2context_vectors['C0000139']['long'].shape[0] == 300


    def test_negative(self):
        cuis = list(self.cdb.cui2names.keys())
        for cui in cuis:
            vectors = {}
            for cntx_type in self.config.linking['context_vector_sizes']:
                vectors[cntx_type] = np.random.rand(300)
            self.cdb.update_context_vector(cui, vectors, negative=True)

        assert self.cdb.cui2count_train['C0000139'] == 2
        assert self.cdb.cui2context_vectors['C0000139']['long'].shape[0] == 300

    def test_save_and_load(self):
        self.cdb.save("./tmp_cdb.dat")
        cdb2 = CDB.load('./tmp_cdb.dat')
        # Check a random thing
        assert cdb2.cui2context_vectors['C0000139']['long'][7] == self.cdb.cui2context_vectors['C0000139']['long'][7]

    def test_training_import(self):
        cdb2 = CDB.load('./tmp_cdb.dat')
        self.cdb.reset_training()
        cdb2.reset_training()
        np.random.seed(11)
        cuis = list(self.cdb.cui2names.keys())
        for i in range(2):
            for cui in cuis:
                vectors = {}
                for cntx_type in self.config.linking['context_vector_sizes']:
                    vectors[cntx_type] = np.random.rand(300)
                self.cdb.update_context_vector(cui, vectors, negative=False)

        cdb2.import_training(cdb=self.cdb, overwrite=True)
        assert cdb2.cui2context_vectors['C0000139']['long'][7] == self.cdb.cui2context_vectors['C0000139']['long'][7]
        assert cdb2.cui2count_train['C0000139'] == self.cdb.cui2count_train['C0000139']

    def test_concept_similarity(self):
        cdb = CDB(config=self.config)
        np.random.seed(11)
        for i in range(500):
            cui = "C" + str(i)
            type_ids = {'T-' + str(i%10)}
            cdb.add_concept(cui=cui, names=prepare_name('Name: ' + str(i), self.maker.pipe.get_spacy_nlp(), {}, self.config), ontologies=set(),
                            name_status='P', type_ids=type_ids, description='', full_build=True)

            vectors = {}
            for cntx_type in self.config.linking['context_vector_sizes']:
                vectors[cntx_type] = np.random.rand(300)
            cdb.update_context_vector(cui, vectors, negative=False)
        res = cdb.most_similar('C200', 'long', type_id_filter=['T-0'], min_cnt=1, topn=10, force_build=True)
        assert len(res) == 10

    def test_training_reset(self):
        self.cdb.reset_training()
        assert len(self.cdb.cui2context_vectors['C0']) == 0
        assert self.cdb.cui2count_train['C0'] == 0
