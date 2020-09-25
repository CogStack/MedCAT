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
from medcat.utils.loggers import basic_logger

log = basic_logger("CAT")

class MakeVocab(object):
    r'''
    Create a new vocab from a text file. To make a vocab and train word embeddings do:
    >>>cdb = <your existing cdb>
    >>>maker = MakeVocab(cdb=cdb)
    >>>maker.make(data_iterator, out_folder="./output/")
    >>>maker.add_vectors(self, in_path="./output/data.txt")
    >>>

    Args:
        cdb (medcat.cdb.CDB):
            The concept database that will be added ontop of the Vocab built from the text file.
        vocab (medcat.utils.vocab.Vocab, optional):
            Vocabulary to be extended, leave as None if you want to make a new Vocab. Default: None
        word_tokenizer (<function>):
            A custom tokenizer for word spliting - used if embeddings are BERT or similar.
            Default: None
    '''

    def __init__(self, cdb=None, vocab=None, word_tokenizer=None):
        self.cdb = cdb
        self.w2v = None
        self.vocab_path = "./vocab.dat"
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocab()

        # Build the required spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all, disable=['ner', 'parser', 'vectors', 'textcat'])

        # Get the tokenizer
        if word_tokenizer is not None:
            self.tokenizer = word_tokenizer
        else:
            self.tokenizer = self._tok


    def _tok(self, text):
        return [text]


    def make(self, iter_data, out_folder, join_cdb=True):
        r'''
        Make a vocab - without vectors initially. This will create two files in the out_folder:
        - vocab.dat -> The vocabulary without vectors
        - data.txt -> The tokenized dataset prepared for training of word2vec or similar embeddings. 

        Args:
            iter_data (Iterator):
                An iterator over sentences or documents. Can also be a simple array of text documents/sentences.
            out_folder (string):
                A path to a folder where all the results will be saved
            join_cdb (bool):
                Should the words from the CDB be added to the Vocab. Default: True
        '''
        # Save the preprocessed data, used for emb training
        out_path = out_folder + "data.txt"
        vocab_path = out_folder + "vocab.dat"
        self.vocab_path = vocab_path
        out = open(out_path, 'w')

        for ind, doc in enumerate(iter_data):
            if ind % 10000 == 0:
                log.info("Vocab builder at: " + str(ind))
                print(ind)

            doc = self.nlp.nlp.tokenizer(doc)
            line = ""

            for token in doc:
                if token.is_space or token.is_punct:
                    continue

                if len(token.lower_) > 0:
                    self.vocab.inc_or_add(token.lower_)

                line = line + " " + "_".join(token.lower_.split(" "))

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


    def add_vectors(self, in_path=None, w2v=None, overwrite=False, data_iter=None, workers=8, niter=2, min_count=10, window=10, vsize=300):
        r'''
        Add vectors to an existing vocabulary and save changes to the vocab_path. 

        Args:
            in_path (String):
                Path to the data.txt that was created by the MakeVocab.make() function.
            w2v (Word2Vec, optional):
                An existing word2vec instance. Default: None
            overwrite (bool):
                If True it will overwrite existing vectors in the vocabulary. Default: False
            data_iter (iterator):
                If you want to provide a customer iterator over the data use this. If yes, then in_path is not needed.
            **: Word2Vec arguments

        Returns:
            A trained word2vec model.
        '''
        if w2v is None:
            if data_iter is None:
                data = SimpleIter(in_path)
            else:
                data = data_iter
            w2v = Word2Vec(data, window=window, min_count=min_count, workers=workers, size=vsize, iter=niter)

        for word in w2v.wv.vocab.keys():
            if word in self.vocab:
                if overwrite:
                    self.vocab.add_vec(word, w2v.wv.get_vector(word))
                else:
                    if self.vocab.vec(word) is None:
                        self.vocab.add_vec(word, w2v.wv.get_vector(word))

        # Save the vocab again, now with vectors
        self.vocab.make_unigram_table()
        self.vocab.save_dict(path=self.vocab_path)
        return w2v
