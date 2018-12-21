from cat.make_word_vectors import WordEmbedding
from preprocessing.iterators import EmbMimicCSV
import pandas
import spacy
from spacy.tokenizer import Tokenizer
from cat.umls import UMLS
from cat.spacy_cat import SpacyCat
from preprocessing.tokenizers import spacy_split_all
from preprocessing.cleaners import spacy_tag_punct, clean_umls
from spacy.tokens import Token
from preprocessing.spelling import CustomSpellChecker, SpacySpellChecker
from preprocessing.spacy_pipe import SpacyPipe
from preprocessing.iterators import EmbMimicCSV
from gensim.models import FastText
from cat.prepare_umls import PrepareUMLS

# Get vocab 
emb_dict = {}
f = open('/home/ubuntu/data/other/dicts/wiki-umls.txt')
for line in f:
    pts = line.strip().split('\t')
    emb_dict[pts[0]] = int(pts[1])

# Build umls
csv_paths = ["/home/ubuntu/data/umls/raw/hpo-plus_concepts.csv", "/home/ubuntu/data/umls/raw/atc-plus_concepts.csv", "/home/ubuntu/data/umls/raw/snomedctus-plus_concepts.txt", "/home/ubuntu/data/umls/raw/icd10pcs-plus_concepts.csv", "/home/ubuntu/data/umls/raw/rxnorm-plus_concepts.csv"]
csv_paths = ['/home/ubuntu/data/umls/raw/hpo-plus_concepts.csv']
prep = PrepareUMLS()
prep.prepare_csvs(csv_paths)
umls = prep.umls
try:
    umls.save("/home/ubuntu/data/umls/models/hpo-plus.dat")
except:
    print("BLABLA")
    pass
#umls = UMLS.load("/tmp/umls.dat")

# Build tokenizer
nlp = SpacyPipe(spacy_split_all)
nlp.add_punct_tagger(tagger=spacy_tag_punct)
spell_checker = CustomSpellChecker(words=umls.vocab, big_vocab=emb_dict)
nlp.add_spell_checker(spell_checker=spell_checker)


out = open("/home/ubuntu/data/mimic/preprocessed/emb_noteevents.txt", 'w')
m_csv_paths = ["/home/ubuntu/data/mimic/raw/noteevents.csv"]
iter_data = EmbMimicCSV(csv_paths=m_csv_paths, tokenizer=nlp, emb_dict=emb_dict)
cnt = 0
for doc in iter_data:
    if cnt % 10000 == 0:
        print("DONE: " + str(cnt))
    cnt += 1

    out.write(" ".join(doc))
    out.write("\n")

class iterator(object):
    def __iter__(self):
        data = open("/home/ubuntu/data/mimic/preprocessed/emb_noteevents.txt", 'r')

        for row in data:
            row = row.strip().split(" ")
            yield row

data = iterator()

we = WordEmbedding()
we.make_vectors(data)

# Save the data

