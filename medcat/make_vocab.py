from medcat.utils.vocab import Vocab
import numpy as np
import pandas
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.preprocessing.cleaners import spacy_tag_punct, clean_name, clean_def
from medcat.utils.spacy_pipe import SpacyPipe
from functools import partial
from medcat.utils.spelling import CustomSpellChecker
from gensim.models import Word2Vec
from medcat.preprocessing.iterators import SimpleIter

class MakeVocab(object):
    SKIP_STOPWORDS = False

    def __init__(self, cdb, vocab=None, word_tokenizer=None):
        self.cdb = cdb

        self.w2v = None
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocab()

        # Build the required spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all, disable=['ner', 'parser', 'vectors', 'textcat'])
        self.nlp.add_punct_tagger(tagger=partial(spacy_tag_punct, skip_stopwords=self.SKIP_STOPWORDS))
        self.spell_checker = CustomSpellChecker(cdb_vocab=cdb.vocab, data_vocab=vocab)
        self.nlp.add_spell_checker(spell_checker=self.spell_checker)

        # Get the tokenizer
        if word_tokenizer is not None:
            self.tokenizer = word_tokenizer
        else:
            self.tokenizer = self._tok

    def _tok(self, text):
        return [text]



    def make(self, iter_data, out_folder, join_cdb=True):
        # Save the preprocessed data, used for emb training
        out_path = out_folder + "data.txt"
        vocab_path = out_folder + "vocab.dat"
        out = open(out_path, 'w')

        for ind, doc in enumerate(iter_data):
            if ind % 10000 == 0:
                print(ind)

            doc = self.nlp(doc)
            line = ""

            for token in doc:
                if token._.to_skip:
                    continue

                if len(token._.norm) > 1:
                    self.vocab.inc_or_add(token._.norm)
                    # Add also the unnormalized version if it is different
                    if token._.norm != token.lower_:
                        self.vocab.inc_or_add(token.lower_)

                line = line + " " + "_".join(token._.norm.split(" "))

            out.write(line.strip())
            out.write("\n")
        out.close()

        if join_cdb and self.cdb:
            for word in self.cdb.vocab.keys():
                if word not in self.vocab:
                    self.vocab.add_word(word)
                else:
                    # Update the count with the counts from the new dataset
                    self.cdb.vocab[word] += self.vocab[word]

        # Save the vocab also
        self.vocab.save_dict(path=vocab_path)


    def add_vectors(self, in_path, overwrite=False):
        data = SimpleIter(in_path)
        w2v = Word2Vec(data, window=10, min_count=10, workers=8, size=300, iter=2)

        for word in w2v.wv.vocab.keys():
            if word in self.vocab:
                if overwrite:
                    self.vocab.add_vec(word, w2v.wv.get_vector(word))
                else:
                    if self.vocab.vec(word) is None:
                        self.vocab.add_vec(word, w2v.wv.get_vector(word))


        return w2v
