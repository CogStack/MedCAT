import unittest
import logging
import os
import numpy as np
from medcat.cdb_maker import CDBMaker
from medcat.cdb import CDB
from medcat.config import Config
from medcat.preprocessing.cleaners import prepare_name

#cdb.csv
#cui  name  ontologies  name_status type_ids  description
#C0000039 Virus MSH P T109|T123 Synthetic phospholipid used in liposomes and lipid bilayers to study biological membranes. It is also a major constituent of PULMONARY SURFACTANTS.
#C0000039 Virus M     T234  
#C0000039 Virus M |Virus K|Virus Z        
#C0000139 Virus M|Virus K|Virus Z   P   
#C0000139 Virus   A   

#cdb2.csv
#cui  name  ontologies  name_status type_ids  description
#C0000239 Second csv        

#TESTS RUN IN ALPHABETICAL ORDER - CONTROLLING WITH '[class_letter]Class and test_[classletter subclassletter]' function syntax


class A_CDBMakerLoadTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Load test database csvs for load tests")
        config = Config()
        config.general['log_level'] = logging.DEBUG
        config.general["spacy_model"] = "en_core_web_md"
        cls.maker = CDBMaker(config)
        csvs = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'examples', 'cdb.csv'),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'examples', 'cdb_2.csv')
        ]
        cls.cdb = cls.maker.prepare_csvs(csvs, full_build=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.maker.destroy_pipe()

    def test_aa_cdb_names_length(self):
        self.assertEqual(len(self.cdb.cui2names), 3, "Should equal 3")

    def test_ab_cdb_names_output(self):
        target_result = {'C0000039': {'virus~k', 'virus', 'virus~m', 'virus~z'}, 'C0000139': {'virus~k', 'virus', 'virus~m', 'virus~z'}, 'C0000239': {'second~csv'}}
        self.assertEqual(self.cdb.cui2names, target_result)

    def test_ac_cdb_snames_length(self):
        self.assertEqual(len(self.cdb.cui2snames), 3, "Should equal 3")

    def test_ad_cdb_snames_output(self):
        target_result = {'C0000039': {'virus~k', 'virus', 'virus~m', 'virus~z'}, 'C0000139': {'virus~k', 'virus', 'virus~m', 'virus~z'}, 'C0000239': {'second', 'second~csv'}}
        self.assertEqual(self.cdb.cui2snames, target_result)

    def test_ae_cdb_name_to_cuis_length(self):
        self.assertEqual(len(self.cdb.name2cuis), 5, "Should equal 5")

    def test_af_cdb_name_to_cuis_output(self):
        target_result = {'virus': ['C0000039', 'C0000139'], 'virus~m': ['C0000039', 'C0000139'], 'virus~k': ['C0000039', 'C0000139'], 'virus~z': ['C0000039', 'C0000139'], 'second~csv': ['C0000239']}
        self.assertEqual(self.cdb.name2cuis, target_result)

    def test_ag_cdb_cuis_to_tags_length(self):
        self.assertEqual(len(self.cdb.cui2tags), 0, "Should equal 0")

    def test_ah_cdb_cuis_to_tags_output(self):
        target_result = {}
        self.assertEqual(self.cdb.cui2tags, target_result)

    def test_ai_cdb_cui_to_preferred_name_length(self):
        self.assertEqual(len(self.cdb.cui2preferred_name), 2, "Should equal 2")

    def test_aj_cdb_cui_to_preferred_name_output(self):
        target_result = {'C0000039': 'Virus', 'C0000139': 'Virus Z'}
        self.assertEqual(self.cdb.cui2preferred_name, target_result)

    def test_ak_cdb_cui_to_context_vectors_length(self):
        self.assertEqual(len(self.cdb.cui2context_vectors), 0, "Should equal 0")

    def test_al_cdb_cui_to_context_vectors_output(self):
        target_result = {}
        self.assertEqual(self.cdb.cui2context_vectors, target_result)

    def test_am_cdb_cui_to_count_train_length(self):
        self.assertEqual(len(self.cdb.cui2count_train), 0, "Should equal 0")

    def test_an_cdb_cui_to_count_train_output(self):
        target_result = {}
        self.assertEqual(self.cdb.cui2count_train, target_result)

    def test_ao_cdb_name_to_cui_to_status_length(self):
        self.assertEqual(len(self.cdb.name2cuis2status), 5, "Should equal 5")

    def test_ap_cdb_name_to_cui_to_status_output(self):
        target_result = {'virus': {'C0000039': 'P', 'C0000139': 'A'}, 'virus~m': {'C0000039': 'A', 'C0000139': 'P'}, 'virus~k': {'C0000039': 'A', 'C0000139': 'P'}, 'virus~z': {'C0000039': 'A', 'C0000139': 'P'}, 'second~csv': {'C0000239': 'A'}}
        self.assertEqual(self.cdb.name2cuis2status, target_result)

    def test_aq_cdb_cui_to_type_ids_length(self):
        self.assertEqual(len(self.cdb.cui2type_ids), 3, "Should equal 3")

    def test_ar_cdb_cui_to_type_ids_output(self):
        target_result = {'C0000039': {'T234', 'T109', 'T123'}, 'C0000139': set(), 'C0000239': set()}
        self.assertEqual(self.cdb.cui2type_ids, target_result)

    def test_as_cdb_additional_info_length(self):
        self.assertEqual(len(self.cdb.addl_info), 8, "Should equal 8")

    def test_at_cdb_additional_info_output(self):
        target_result = {'cui2icd10': {}, 'cui2opcs4': {}, 'cui2ontologies': {'C0000039': {'MSH'}}, 'cui2original_names': {'C0000039': {'Virus K', 'Virus M', 'Virus', 'Virus Z'}, 'C0000139': {'Virus K', 'Virus M', 'Virus', 'Virus Z'}, 'C0000239': {'Second csv'}}, 'cui2description': {'C0000039': 'Synthetic phospholipid used in liposomes and lipid bilayers to study biological membranes. It is also a major constituent of PULMONARY SURFACTANTS.'}, 'type_id2name': {}, 'type_id2cuis': {'T109': {'C0000039'}, 'T123': {'C0000039'}, 'T234': {'C0000039'}}, 'cui2group': {}}
        self.assertEqual(self.cdb.addl_info, target_result)


class B_CDBMakerEditTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Load test database csvs for edit tests")
        cls.config = Config()
        cls.config.general['log_level'] = logging.DEBUG
        cls.config.general["spacy_model"] = "en_core_web_md"
        cls.maker = CDBMaker(cls.config)
        csvs = [
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'examples', 'cdb.csv'),
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'examples', 'cdb_2.csv')
        ]
        cls.cdb = cls.maker.prepare_csvs(csvs, full_build=True)
        cls.cdb2 = CDB(cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.maker.destroy_pipe()

    def test_ba_addition_of_new_name(self):
        self.cdb.add_names(cui='C0000239', names=prepare_name('MY: new,-_! Name.', self.maker.pipe.spacy_nlp, {}, self.config), name_status='P', full_build=True)
        self.assertEqual(len(self.cdb.name2cuis), 6, "Should equal 6")
        target_result = {'MY: new,-_! Name.', 'Second csv'}
        self.assertEqual(self.cdb.addl_info['cui2original_names']['C0000239'], target_result)
        self.assertIn('my~:~new~name~.', self.cdb.name2cuis)
        self.assertIn('my~:~new', self.cdb.snames)
        self.assertIn('my~:~new~name~.', self.cdb.name2cuis2status)

    def test_bb_removal_of_name(self):
        self.cdb.remove_names(cui='C0000239', names=prepare_name('MY: new,-_! Name.', self.maker.pipe.spacy_nlp, {}, self.config))
        self.assertEqual(len(self.cdb.name2cuis), 5, "Should equal 5")
        self.assertNotIn('my:newname.', self.cdb.name2cuis2status)

    def test_bc_filter_by_cui(self):
        cuis_to_keep = {'C0000039'}
        self.cdb.filter_by_cui(cuis_to_keep=cuis_to_keep)
        self.assertEqual(len(self.cdb.cui2names), 2, "Should equal 2")
        self.assertEqual(len(self.cdb.name2cuis), 4, "Should equal 4")
        self.assertEqual(len(self.cdb.snames), 4, "Should equal 4")

    def test_bd_addition_of_context_vector_positive(self):
        np.random.seed(11)
        cuis = list(self.cdb.cui2names.keys())
        for i in range(2):
                for cui in cuis:
                        vectors = {}
                        for cntx_type in self.config.linking['context_vector_sizes']:
                                vectors[cntx_type] = np.random.rand(300)
                        self.cdb.update_context_vector(cui, vectors, negative=False)

        self.assertEqual(self.cdb.cui2count_train['C0000139'], 2, "Count should equal 2")
        self.assertEqual(self.cdb.cui2context_vectors['C0000139']['long'].shape[0], 300, "Dimensions should equal 300")

    def test_be_addition_of_context_vector_negative(self):
        np.random.seed(11)
        cuis = list(self.cdb.cui2names.keys())
        for i in range(2):
                for cui in cuis:
                        vectors = {}
                        for cntx_type in self.config.linking['context_vector_sizes']:
                                vectors[cntx_type] = np.random.rand(300)
                        self.cdb.update_context_vector(cui, vectors, negative=True)

        self.assertEqual(self.cdb.cui2count_train['C0000139'], 2, "Count should equal 2")
        self.assertEqual(self.cdb.cui2context_vectors['C0000139']['long'].shape[0], 300, "Dimensions should equal 300")

    def test_bf_import_training(self):
        self.cdb2.import_training(cdb=self.cdb, overwrite=True)
        self.assertEqual(self.cdb.cui2count_train['C0000139'], 2, "Count should equal 2")
        self.assertEqual(self.cdb.cui2context_vectors['C0000139']['long'].shape[0], 300, "Dimensions should equal 300")


    def test_bg_save_and_load_model_context_vectors(self):
        self.cdb.save("./tmp_cdb.dat")
        self.cdb2 = CDB.load('./tmp_cdb.dat')
        self.assertEqual(self.cdb.cui2count_train['C0000139'], 2, "Count should equal 2")
        self.assertEqual(self.cdb.cui2context_vectors['C0000139']['long'].shape[0], 300, "Dimensions should equal 300")


    def test_bh_reset_training(self):
        self.cdb.reset_training()
        target_result = {}
        self.assertEqual(self.cdb.cui2context_vectors, target_result)


if __name__ == '__main__':
    unittest.main()
