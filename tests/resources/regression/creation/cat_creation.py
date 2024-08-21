import os
import sys
import pandas as pd

from medcat.vocab import Vocab
from medcat.config import Config
from medcat.cdb_maker import CDBMaker
from medcat.cdb import CDB
from medcat.cat import CAT


vi = sys.version_info
PY_VER = f"{vi.major}.{vi.minor}"


# paths
VOCAB_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'vocab_data.txt'
    # os.path.dirname(__file__), 'vocab_data_auto.txt'
)
CDB_PREPROCESSED_PATH = os.path.join(
    os.path.dirname(__file__), 'preprocessed4cdb.txt'
)
SELF_SUPERVISED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'selfsupervised_data.txt'
)
SUPERVISED_DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'supervised_mct_export.json'
)
SAVE_PATH = os.path.dirname(__file__)
SAVE_NAME = f"simple_model4test-{PY_VER}"

# vocab

vocab = Vocab()
vocab.add_words(VOCAB_DATA_PATH)

# CDB
config = Config()

maker = CDBMaker(config)

cdb: CDB = maker.prepare_csvs([CDB_PREPROCESSED_PATH])

# CAT
cat = CAT(cdb, vocab)

# training
# self-supervised
data = pd.read_csv(SELF_SUPERVISED_DATA_PATH)
cat.train(data.text.values)

print("[sst] cui2count_train", cat.cdb.cui2count_train)

# supervised

cat.train_supervised_from_json(SUPERVISED_DATA_PATH)

print("[sup] cui2count_train", cat.cdb.cui2count_train)

# save
mpn = cat.create_model_pack(SAVE_PATH, model_pack_name=SAVE_NAME)
full_path = os.path.join(SAVE_PATH, mpn)
print("Saved to")
print(full_path)
