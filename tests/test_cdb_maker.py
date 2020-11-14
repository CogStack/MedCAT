%load_ext autoreload
%autoreload 2

from medcat.cdb_maker import CDBMaker
from medcat.config import Config
import numpy as np
import logging

config = Config()
config.general['log_level'] = logging.INFO
maker = CDBMaker(config)


csvs = ['/home/ubuntu/data/snomed/test.csv']
csvs = ['/home/ubuntu/data/snomed/snomed_cdb_csv_SNOMED-CT-full_UK_drug_ext_Release_20200228.csv']
#csvs = ['/home/ubuntu/data/snomed/umls_names_extension.csv']

cdb = maker.prepare_csvs(csvs)

# Test adding vectors for concepts
cuis = list(cdb.cui2names.keys())
for cui in cuis[0:50]:
    vectors = {'short': np.random.rand(300),
              'long': np.random.rand(300)
              }
    cdb.update_context_vector(cui, vectors, negative=False)

cuis_to_keep = ['S-478006', 'S-476005', 'S-486006', 'S-487002']
cdb.filter_by_cui(cuis_to_keep=cuis_to_keep)
