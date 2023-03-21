import logging
from pathlib import Path
from gensim.models import Word2Vec
from medcat.vocab import Vocab
from medcat.pipe import Pipe
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.preprocessing.iterators import SimpleIter
from medcat.preprocessing.taggers import tag_skip_and_punct


logger = logging.getLogger(__name__)


class MakeVocab(object):
    """Create a new vocab from a text file.

    Args:
        config (medcat.config.Config):
            Global configuration for medcat.
        cdb (medcat.cdb.CDB):
            The concept database that will be added ontop of the Vocab built from the text file.
        vocab (medcat.vocab.Vocab, optional):
            Vocabulary to be extended, leave as None if you want to make a new Vocab. Default: None
        word_tokenizer (<function>):
            A custom tokenizer for word spliting - used if embeddings are BERT or similar.
            Default: None
    Examples:
        To make a vocab and train word embeddings.

        >>> cdb = <your existing cdb>
        >>> maker = MakeVocab(cdb=cdb, config=config)
        >>> maker.make(data_iterator, out_folder="./output/")
        >>> maker.add_vectors(in_path="./output/data.txt")
    """

    def __init__(self, config, cdb=None, vocab=None, word_tokenizer=None):
        self.cdb = cdb
        self.config = config
        self.w2v = None
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocab()

        # Build the required spacy pipeline
        self.pipe = Pipe(tokenizer=spacy_split_all, config=config)
        self.pipe.add_tagger(tagger=tag_skip_and_punct,
                             name='skip_and_punct',
                             additional_fields=['is_punct'])

        # Get the tokenizer
        if word_tokenizer is not None:
            self.tokenizer = word_tokenizer
        else:
            self.tokenizer = self._tok

        # Used for saving if the real path is not set
        self.vocab_path = "./tmp_vocab.dat"

    def _tok(self, text):
        return [text]

    def make(self, iter_data, out_folder, join_cdb=True, normalize_tokens=False):
        """Make a vocab - without vectors initially. This will create two files in the out_folder:
        - vocab.dat -> The vocabulary without vectors
        - data.txt -> The tokenized dataset prepared for training of word2vec or similar embeddings.

        Args:
            iter_data (Iterator):
                An iterator over sentences or documents. Can also be a simple array of text documents/sentences.
            out_folder (string):
                A path to a folder where all the results will be saved.
            join_cdb (bool):
                Should the words from the CDB be added to the Vocab. Default: True.
            normalize_tokens (bool, defaults to True):
                If set tokens will be lematized - tends to work better in some cases where the difference
                between e.g. plural/singular should be ignored. But in general not so important if the dataset is big enough.
        """
        # Save the preprocessed data, used for emb training
        out_path = Path(out_folder) / "data.txt"
        vocab_path = Path(out_folder) / "vocab.dat"
        self.vocab_path = vocab_path
        out = open(out_path, 'w', encoding='utf-8')

        for ind, doc in enumerate(iter_data):
            if ind % 10000 == 0:
                logger.info("Vocab builder at: %s", str(ind))

            doc = self.pipe.spacy_nlp.tokenizer(doc)
            line = ""

            for token in doc:
                if token.is_space or token.is_punct:
                    continue

                if len(token.lower_) > 0:
                    if normalize_tokens:
                        self.vocab.inc_or_add(token._.norm)
                    else:
                        self.vocab.inc_or_add(token.lower_)

                if normalize_tokens:
                    line = line + " " + "_".join(token._.norm.split(" "))
                else:
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
        self.vocab.save(path=self.vocab_path)

    def add_vectors(self, in_path=None, w2v=None, overwrite=False, data_iter=None, workers=14, epochs=2, min_count=10, window=10, vector_size=300,
                    unigram_table_size=100000000):
        """Add vectors to an existing vocabulary and save changes to the vocab_path.

        Args:
            in_path (str):
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
        """
        if w2v is None:
            if data_iter is None:
                data = SimpleIter(in_path)
            else:
                data = data_iter
            w2v = Word2Vec(data, window=window, min_count=min_count, workers=workers, vector_size=vector_size, epochs=epochs)

        for word in w2v.wv.key_to_index.keys():
            if word in self.vocab:
                if overwrite:
                    self.vocab.add_vec(word, w2v.wv.get_vector(word))
                else:
                    if self.vocab.vec(word) is None:
                        self.vocab.add_vec(word, w2v.wv.get_vector(word))

        # Save the vocab again, now with vectors
        self.vocab.make_unigram_table(table_size=unigram_table_size)
        self.vocab.save(path=self.vocab_path)
        return w2v

    def destroy_pipe(self):
        self.pipe.destroy()
