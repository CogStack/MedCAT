from medcat.cdb_maker import CDBMaker
from medcat.config import Config
import numpy as np
import logging
import os

config = Config()
config.general['log_level'] = logging.INFO
config.general['spacy_model'] = 'en_core_sci_lg'
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

cdb.reset_training()

cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)
cat.config.ner['min_name_len'] = 3
cat.config.ner['upper_case_limit_len'] = 3
cat.config.linking['disamb_length_limit'] = 3
cat.config.linking['filters'] = {'cuis': set()}
cat.config.linking['train_count_threshold'] = -1
cat.config.linking['context_vector_sizes'] = {'xlong': 27, 'long': 18, 'medium': 9, 'short': 3}
cat.config.linking['context_vector_weights'] = {'xlong': 0, 'long': 0.4, 'medium': 0.4, 'short': 0.2}
cat.config.linking['weighted_average_function'] = lambda step: max(0.1, 1-(step**2*0.0004))

cat.config.linking['similarity_threshold_type'] = 'dynamic'
cat.config.linking['similarity_threshold'] = 0.35
cat.config.linking['calculate_dynamic_threshold'] = True

cat.train(df.text.values, fine_tune=True)


cdb.config.general['spacy_disabled_components'] = ['ner', 'parser', 'vectors', 'textcat',
                                                      'entity_linker', 'sentencizer', 'entity_ruler', 'merge_noun_chunks',
                                                                                                    'merge_entities', 'merge_subtokens']

%load_ext autoreload
%autoreload 2

# Train
_ = cat.train(open("./tmp_medmentions_text_only.txt", 'r'), fine_tune=False)

_ = cat.train_supervised("/home/ubuntu/data/medmentions/medmentions.json", reset_cui_count=True, nepochs=13, train_from_false_positives=True, print_stats=3, test_size=0)
cdb.save("/home/ubuntu/data/umls/2020ab/cdb_trained_medmen.dat")


_ = cat.train_supervised("/home/ubuntu/data/medmentions/medmentions.json", reset_cui_count=True, nepochs=13, train_from_false_positives=True, print_stats=0, test_size=0)

cat.config.linking['similarity_threshold'] = 0.1
cat.config.ner['min_name_len'] = 2
cat.config.ner['upper_case_limit_len'] = 1
cat.config.linking['train_count_threshold'] = -2
cat.config.linking['filters']['cuis'] = set()
cat.config.linking['context_vector_sizes'] = {'xlong': 27, 'long': 18, 'medium': 9, 'short': 3}
cat.config.linking['context_vector_weights'] = {'xlong': 0.1, 'long': 0.4, 'medium': 0.4, 'short': 0.1}

cat.config.linking['similarity_threshold_type'] = 'dynamic'
cat.config.linking['similarity_threshold'] = 0.35
cat.config.linking['calculate_dynamic_threshold'] = True


# Print some stats
_ = cat._print_stats(data)

#static:  Epoch: 0, Prec: 0.4182515298479039, Rec: 0.5144124168514412, F1: 0.461374712901238 
#dynamic: Epoch: 0, Prec: 0.41783918321735153, Rec: 0.5137878267387027, F1: 0.4608726101267596
#d wrat : Epoch: 0, Prec: 0.42184109928711644, Rec: 0.5100402860622716, F1: 0.4617668264133339
#d rem: : Epoch: 0, Prec: 0.42636048572713564, Rec: 0.5032789956904629, F1: 0.46163761618997146  
#b7       Epoch: 0, Prec: 0.42293654517962354, Rec: 0.5066204484416963, F1: 0.46101165103722647
#wnumb  : Epoch: 0, Prec: 0.4293618687198225, Rec: 0.5137565972330658, F1: 0.467783212010919
#6k       Epoch: 0, Prec: 0.43320010658140157, Rec: 0.5077293026451392, F1: 0.46751304797918075
#6k 5     Epoch: 0, Prec: 0.4325832297643917, Rec: 0.5108834827144686, F1: 0.4684842063060225
# 420, 498, 456
# p: 0.413  r: 0.505  f1: 0.454

# Epoch: 0, Prec: 0.4281473766436662, Rec: 0.5135067611879703, F1: 0.46695822565529777


# Remove all names that are numbers
for name in list(cdb.name2cuis.keys()):
    if name.replace(".", '').replace("~", '').replace(",", '').replace(":", '').replace("-", '').isnumeric():
        del cdb.name2cuis[name]
        print(name)


for name in list(cdb.name2cuis.keys()):
    if len(name) < 7 and (not name.isalpha()) and len(re.sub("[^A-Za-z]*", '', name)) < 2:
        del cdb.name2cuis[name]
        print(name)




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


cdb = maker.prepare_csvs(csv_paths=csvs)
cdb.save("/home/ubuntu/data/umls/2020ab/cdb_vbg.dat")
