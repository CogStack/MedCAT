r''' The tests here are a bit messy but they work, should be converted to python unittests.
'''
from medcat.cdb_maker import CDBMaker
from medcat.config import Config
import numpy as np
import logging

config = Config()
config.general['log_level'] = logging.DEBUG
maker = CDBMaker(config)

# Building a new CDB from two files (full_build)
csvs = ['../examples/cdb.csv', '../examples/cdb_2.csv']
cdb = maker.prepare_csvs(csvs, full_build=True)

assert len(cdb.cui2names) == 3
assert len(cdb.cui2snames) == 3
assert len(cdb.name2cuis) == 5
assert len(cdb.cui2tags) == 3
assert len(cdb.cui2preferred_name) == 2
assert len(cdb.cui2context_vectors) == 3
assert len(cdb.cui2count_train) == 3
assert cdb.name2cuis2status['virus']['C0000039'] == 'P'
assert cdb.cui2type_ids['C0000039'] == {'T234', 'T109', 'T123'}
assert cdb.addl_info['cui2original_names']['C0000039'] == {'Virus', 'Virus K', 'Virus M', 'Virus Z'}
assert cdb.addl_info['cui2description']['C0000039'].startswith("Synthetic")

# Test name addition
from medcat.preprocessing.cleaners import prepare_name
cdb.add_names(cui='C0000239', names=prepare_name('MY: new,-_! Name.', maker.nlp, {}, config), name_status='P', full_build=True)
assert cdb.addl_info['cui2original_names']['C0000239'] == {'MY: new,-_! Name.', 'Second csv'}
assert 'my:newname.' in cdb.name2cuis
assert 'my:new' in cdb.snames
assert 'my:newname.' in cdb.name2cuis2status
assert cdb.name2cuis2status['my:newname.'] == {'C0000239': 'P'}

# Test name removal
cdb.remove_names(cui='C0000239', names=prepare_name('MY: new,-_! Name.', maker.nlp, {}, config))
# Run again to make sure it does not break anything
cdb.remove_names(cui='C0000239', names=prepare_name('MY: new,-_! Name.', maker.nlp, {}, config))
assert len(cdb.name2cuis) == 5
assert 'my:newname.' not in cdb.name2cuis2status

# Test filtering
cuis_to_keep = {'C0000039'} # Because of transition 2 will be kept
cdb.filter_by_cui(cuis_to_keep=cuis_to_keep)
assert len(cdb.cui2names) == 2
assert len(cdb.name2cuis) == 4
assert len(cdb.snames) == 4

# Test vector addition
import numpy as np
cdb.reset_training()
np.random.seed(11)
cuis = list(cdb.cui2names.keys())
for i in range(2):
    for cui in cuis:
        vectors = {}
        for cntx_type in config.linking['context_vector_sizes']:
            vectors[cntx_type] = np.random.rand(300)
        cdb.update_context_vector(cui, vectors, negative=False)

assert cdb.cui2count_train['C0000139'] == 2
assert cdb.cui2context_vectors['C0000139']['long'].shape[0] == 300
np.testing.assert_approx_equal(np.average(cdb.cui2context_vectors['C0000139']['long']), 0.50, significant=2)


# Test negative
for cui in cuis:
    vectors = {}
    for cntx_type in config.linking['context_vector_sizes']:
        vectors[cntx_type] = np.random.rand(300)
    cdb.update_context_vector(cui, vectors, negative=True)

np.testing.assert_approx_equal(np.average(cdb.cui2context_vectors['C0000139']['long']), 0.23, significant=2)
assert cdb.cui2count_train['C0000139'] == 2
assert cdb.cui2context_vectors['C0000139']['long'].shape[0] == 300

# Test save/load
from medcat.cdb import CDB
cdb.save("./tmp_cdb.dat")
cdb2 = CDB(config=config)
cdb2.load("./tmp_cdb.dat")
# Check a random thing
assert cdb2.cui2context_vectors['C0000139']['long'][7] == cdb.cui2context_vectors['C0000139']['long'][7]

# Test training import
cdb.reset_training()
cdb2.reset_training()
np.random.seed(11)
cuis = list(cdb.cui2names.keys())
for i in range(2):
    for cui in cuis:
        vectors = {}
        for cntx_type in config.linking['context_vector_sizes']:
            vectors[cntx_type] = np.random.rand(300)
        cdb.update_context_vector(cui, vectors, negative=False)

cdb2.import_training(cdb=cdb, overwrite=True)
assert cdb2.cui2context_vectors['C0000139']['long'][7] == cdb.cui2context_vectors['C0000139']['long'][7]
assert cdb2.cui2count_train['C0000139'] == cdb.cui2count_train['C0000139']

# Test concept similarity
cdb = CDB(config=config)
np.random.seed(11)
for i in range(500):
    cui = "C" + str(i)
    type_ids = {'T-' + str(i%10)}
    cdb.add_concept(cui=cui, names=prepare_name('Name: ' + str(i), maker.nlp, {}, config), ontologies=set(),
            name_status='P', type_ids=type_ids, description='', full_build=True)

    vectors = {}
    for cntx_type in config.linking['context_vector_sizes']:
        vectors[cntx_type] = np.random.rand(300)
    cdb.update_context_vector(cui, vectors, negative=False)
res = cdb.most_similar('C200', 'long', type_id_filter=['T-0'], min_cnt=1, topn=10, force_build=True)
assert len(res) == 10

# Test training reset
cdb.reset_training()
assert len(cdb.cui2context_vectors['C0']) == 0
assert cdb.cui2count_train['C0'] == 0
