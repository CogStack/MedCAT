from medcat.cdb_maker import CDBMaker
from medcat.config import Config, weighted_average
from functools import partial
import numpy as np
import logging

from ..helper import VocabDownloader


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

downloader = VocabDownloader()
vocab_path = downloader.vocab_path
downloader.check_or_download()

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
cat.config.linking['weighted_average_function'] = partial(weighted_average, factor=0.0004)
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

_ = cat.train_supervised("/home/ubuntu/data/medmentions/medmentions.json", reset_cui_count=True, nepochs=13, train_from_false_positives=True, print_stats=3, test_size=0.1)
cdb.save("/home/ubuntu/data/umls/2020ab/cdb_trained_medmen.dat")


_ = cat.train_supervised("/home/ubuntu/data/medmentions/medmentions.json", reset_cui_count=False, nepochs=13, train_from_false_positives=True, print_stats=3, test_size=0)

cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)
cat.config.linking['similarity_threshold'] = 0.1
cat.config.ner['min_name_len'] = 2
cat.config.ner['upper_case_limit_len'] = 1
cat.config.linking['train_count_threshold'] = -2
cat.config.linking['filters']['cuis'] = set()
cat.config.linking['context_vector_sizes'] = {'xlong': 27, 'long': 18, 'medium': 9, 'short': 3}
cat.config.linking['context_vector_weights'] = {'xlong': 0.1, 'long': 0.4, 'medium': 0.4, 'short': 0.1}
cat.config.linking['similarity_threshold_type'] = 'static'

cat.config.linking['similarity_threshold_type'] = 'dynamic'
cat.config.linking['similarity_threshold'] = 0.35
cat.config.linking['calculate_dynamic_threshold'] = True


# Print some stats
_ = cat._print_stats(data)

#Epoch: 0, Prec: 0.4331506351144245, Rec: 0.5207520064957372, F1: 0.47292889758643175
#p: 0.421  r: 0.507  f1: 0.460


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
