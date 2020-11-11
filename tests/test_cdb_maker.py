%load_ext autoreload
%autoreload 2

from medcat.cdb_maker import CDBMaker
from medcat.config import Config

config = Config()
maker = CDBMaker(config)

csvs = ['/home/ubuntu/data/snomed/test.csv']
csvs = ['/home/ubuntu/data/snomed/snomed_cdb_csv_SNOMED-CT-full_UK_drug_ext_Release_20200228.csv']
csvs = ['/home/ubuntu/data/snomed/umls_names_extension.csv']

cdb = maker.prepare_csvs(csvs)
