from cat.make_word_vectors import WordEmbedding
import pandas
import spacy
from spacy.tokenizer import Tokenizer
from cat.umls import UMLS
from cat.spacy_cat import SpacyCat
from cat.preprocessing.tokenizers import spacy_split_all
from cat.preprocessing.cleaners import spacy_tag_punct, clean_umls
from spacy.tokens import Token
from cat.preprocessing.spelling import CustomSpellChecker, SpacySpellChecker
from cat.preprocessing.spacy_pipe import SpacyPipe
from cat.preprocessing.iterators import BertEmbMimicCSV
from gensim.models import FastText
from cat.prepare_umls import PrepareUMLS
from cat.preprocessing.vocab import Vocab

# Get vocab 
emb_dict = {}
f = open('/home/ubuntu/data/other/dicts/wiki-umls.txt')
for line in f:
    pts = line.strip().split('\t')
    emb_dict[pts[0]] = int(pts[1])

umls = UMLS()
umls.load_dict('/home/ubuntu/data/umls/models/min-umls-dict-7.dat')

# Build tokenizer
nlp = SpacyPipe(spacy_split_all)
nlp.add_punct_tagger(tagger=spacy_tag_punct)
spell_checker = CustomSpellChecker(words=umls.vocab, big_vocab=emb_dict)
nlp.add_spell_checker(spell_checker=spell_checker)

out = open("/home/ubuntu/data/mimic/preprocessed/emb_noteevents.txt", 'w')
m_csv_paths = ["/home/ubuntu/data/mimic/raw/noteevents.csv"]
iter_data = BertEmbMimicCSV(csv_paths=m_csv_paths, tokenizer=nlp)
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
f = open("/home/ubuntu/data/other/dicts/bert-cnotes-wiki-umls-emb.dat", 'w')
for word in we.model.wv.vocab.keys():
    f.write("{}\t{}\t{}\n".format(word, we.model.wv.vocab[word].count, " ".join([str(x) for x in we.model.wv[word]])))
f.close()


vocab = Vocab()
vocab.add_words("/home/ubuntu/data/other/dicts/bert-cnotes-wiki-umls-emb.dat")
vocab.add_words_nvec("/home/ubuntu/data/other/dicts/wiki-umls.txt")
vocab.save_dict("/home/ubuntu/data/other/dicts/bert-umls-wiki-emb-dict.dat")
