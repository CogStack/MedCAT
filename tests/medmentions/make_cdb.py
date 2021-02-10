from medcat.cdb_maker import CDBMaker
from medcat.config import Config
import numpy as np
import logging
import os

config = Config()
config.general['log_level'] = logging.INFO
maker = CDBMaker(config)

# Building a new CDB from two files (full_build)
csvs = ['./tmp_medmentions.csv']
cdb = maker.prepare_csvs(csvs, full_build=True)

cdb.save("./tmp_cdb.dat")


from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT

vocab_path = "./tmp_vocab.dat"
if not os.path.exists(vocab_path):
    import requests
    tmp = requests.get("https://s3-eu-west-1.amazonaws.com/zkcl/vocab.dat")
    with open(vocab_path, 'wb') as f:
        f.write(tmp.content)

config = Config()
cdb = CDB.load("./tmp_cdb.dat", config=config)
vocab = Vocab.load(vocab_path)
cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)

# Train
_ = cat.train(open("./tmp_medmentions_text_only.txt", 'r'), fine_tune=False)

# Print some stats
_ = cat._print_stats(json.load(open("tmp_test.csv")))


# RUN SUPER
cdb = CDB.load("./tmp_cdb.dat")
vocab = Vocab.load(vocab_path)
cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)

# Train supervised
cdb.reset_cui_count()
cat.config.ner['uppe_case_limit_len'] = 1
cat.config.ner['min_name_len'] = 1
data_path = "./tmp_medmentions.json"
_ = cat.train_supervised(data_path, use_cui_doc_limit=True, nepochs=30, devalue_others=True, test_size=0.2)
