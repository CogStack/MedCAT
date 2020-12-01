from medcat.cdb_maker import CDBMaker
from medcat.config import Config
import numpy as np
import logging

config = Config()
config.general['log_level'] = logging.INFO
maker = CDBMaker(config)

# Building a new CDB from two files (full_build)
csvs = ['./tmp_medmentions.csv']
cdb = maker.prepare_csvs(csvs, full_build=True)

cdb.save("./tmp_cdb.dat")


from medcat.vocab import Vocab
from medcat.cdb import CDB

cdb = CDB.load("./tmp_cdb.dat")
vocab = Vocab()
vocab.load_dict("/home/ubuntu/data/vocabs/vocab.dat")

cat = CAT(cdb, cdb.config, vocab)
