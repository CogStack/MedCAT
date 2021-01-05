import unittest
from medcat.cdb_maker import CDBMaker
from medcat.config import Config
import numpy as np
import logging

#Prevents overhead of loading csvs for every single individual test with unit test's "setUP"
print("Load test database outside of unittest setUp function")
config = Config()
config.general['log_level'] = logging.DEBUG
maker = CDBMaker(config)

#cdb.csv
#cui	name	ontologies	name_status	type_ids	description
#C0000039	Virus	MSH	P	T109|T123	Synthetic phospholipid used in liposomes and lipid bilayers to study biological membranes. It is also a major constituent of PULMONARY SURFACTANTS.
#C0000039	Virus M			T234	
#C0000039	Virus M |Virus K|Virus Z				
#C0000139	Virus M|Virus K|Virus Z		P		
#C0000139	Virus		A		

#cdb2.csv
#cui	name	ontologies	name_status	type_ids	description
#C0000239	Second csv				

# Building a new CDB from two files (full_build)
csvs = ['../examples/cdb.csv', '../examples/cdb_2.csv']
cdb = maker.prepare_csvs(csvs, full_build=True)

class CDBMakerLoadTests(unittest.TestCase):
	#CLARIFY ORDER THAT TESTS APPEAR TO RUN / LOAD IN TERMINAL

	# def setUp(self):
	# 	self.config = Config()
	# 	self.config.general['log_level'] = logging.DEBUG
	# 	self.maker = CDBMaker(self.config)

	# 	# Building a new CDB from two files (full_build)
	# 	csvs = ['../examples/cdb.csv', '../examples/cdb_2.csv']
	# 	self.cdb = self.maker.prepare_csvs(csvs, full_build=True)

	def test_cdb_names_length(self):
		self.assertEqual(len(cdb.cui2names), 3, "Should equal 3")

	def test_cdb_names_output(self):
		target_result = {'C0000039': {'virusz', 'virusk', 'virus', 'virusm'}, 'C0000139': {'virusk', 'virus', 'virusz', 'virusm'}, 'C0000239': {'secondcsv'}}
		self.assertEqual(cdb.cui2names, target_result)

	def test_cdb_snames_length(self):
		self.assertEqual(len(cdb.cui2snames), 3, "Should equal 3")

	def test_cdb_snames_output(self):
		#CLARIFY HOW THIS OUTPUT SHOULD BE DIFFERENT - cui to "sub names"?
		target_result = {'C0000039': {'virusm', 'virusz', 'virusk', 'virus'}, 'C0000139': {'virusm', 'virusz', 'virusk', 'virus'}, 'C0000239': {'second', 'secondcsv'}}
		self.assertEqual(cdb.cui2snames, target_result)

	def test_cdb_name_to_cuis_length(self):
		self.assertEqual(len(cdb.name2cuis), 5, "Should equal 5")

	def test_cdb_name_to_cuis_output(self):
		target_result = {'virus': ['C0000039', 'C0000139'], 'virusm': ['C0000039', 'C0000139'], 'virusk': ['C0000039', 'C0000139'], 'virusz': ['C0000039', 'C0000139'], 'secondcsv': ['C0000239']}
		self.assertEqual(cdb.name2cuis, target_result)

	def test_cdb_cuis_to_tags_length(self):
		self.assertEqual(len(cdb.cui2tags), 3, "Should equal 3")

	def test_cdb_cuis_to_tags_output(self):
		#CONSIDER ADDING TAGS TO DB SO ACTUALLY TESTING CORRECT TRANSFORMATION
		target_result = {'C0000039': [], 'C0000139': [], 'C0000239': []}
		self.assertEqual(cdb.cui2tags, target_result)

	def test_cdb_cui_to_preferred_name_length(self):
		self.assertEqual(len(cdb.cui2preferred_name), 2, "Should equal 2")

	def test_cdb_cui_to_preferred_name_output(self):
		#CLARIFY HOW DECIDED 'VIRUS Z' IS PREFERRED NAME FROM 'Virus M|Virus K|Virus Z'
		target_result = {'C0000039': 'Virus', 'C0000139': 'Virus Z'}
		self.assertEqual(cdb.cui2preferred_name, target_result)

	def test_cdb_cui_to_context_vectors_length(self):
		self.assertEqual(len(cdb.cui2context_vectors), 3, "Should equal 3")

	def test_cdb_cui_to_context_vectors_output(self):
		#CONSIDER ADDING CONTEXT VECTORS TO DB SO ACTUALLY TESTING CORRECT TRANSFORMATION
		target_result = {'C0000039': {}, 'C0000139': {}, 'C0000239': {}}
		self.assertEqual(cdb.cui2context_vectors, target_result)

	def test_cdb_cui_to_count_train_length(self):
		self.assertEqual(len(cdb.cui2count_train), 3, "Should equal 3")

	def test_cdb_cui_to_count_train_output(self):
		#CONSIDER ADDING TRAINING EXAMPLES TO DB SO ACTUALLY TESTING CORRECT TRANSFORMATION
		target_result = {'C0000039': 0, 'C0000139': 0, 'C0000239': 0}
		self.assertEqual(cdb.cui2count_train, target_result)

	def test_cdb_name_to_cui_to_status_length(self):
		self.assertEqual(len(cdb.name2cuis2status), 5, "Should equal 5")

	def test_cdb_name_to_cui_to_status_output(self):
		target_result = {'virus': {'C0000039': 'P', 'C0000139': 'A'}, 'virusm': {'C0000039': 'A', 'C0000139': 'P'}, 'virusk': {'C0000039': 'A', 'C0000139': 'P'}, 'virusz': {'C0000039': 'A', 'C0000139': 'P'}, 'secondcsv': {'C0000239': 'A'}}
		self.assertEqual(cdb.name2cuis2status, target_result)

	def test_cdb_cui_to_type_ids_length(self):
		self.assertEqual(len(cdb.cui2type_ids), 3, "Should equal 3")

	def test_cdb_cui_to_type_ids_output(self):
		target_result = {'C0000039': {'T234', 'T109', 'T123'}, 'C0000139': set(), 'C0000239': set()}
		self.assertEqual(cdb.cui2type_ids, target_result)

	def test_cdb_additional_info_length(self):
		#CLARIFY IF THIS WOULD BE CONSISTENT
		self.assertEqual(len(cdb.addl_info), 8, "Should equal 8")

	def test_cdb_additional_info_output(self):
		#CLARIFY WHY NO DATA ON 139 or 239
		target_result = {'cui2icd10': {}, 'cui2opcs4': {}, 'cui2ontologies': {'C0000039': {'MSH'}}, 'cui2original_names': {'C0000039': {'Virus K', 'Virus M', 'Virus', 'Virus Z'}, 'C0000139': {'Virus K', 'Virus M', 'Virus', 'Virus Z'}, 'C0000239': {'Second csv'}}, 'cui2description': {'C0000039': 'Synthetic phospholipid used in liposomes and lipid bilayers to study biological membranes. It is also a major constituent of PULMONARY SURFACTANTS.'}, 'type_id2name': {}, 'type_id2cuis': {'T109': {'C0000039'}, 'T123': {'C0000039'}, 'T234': {'C0000039'}}, 'cui2group': {}}
		self.assertEqual(cdb.addl_info, target_result)


if __name__ == '__main__':
	unittest.main()