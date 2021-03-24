import numpy as np
import pickle

class Vocab(object):
    r''' Vocabulary used to store word embeddings for context similarity
    calculation. Also used by the spell checker - but not for fixing the spelling
    only for checking is something correct.

    Properties:
        vocab (dict):
            Map from word to attributes, e.g. {'house': {'vec': <np.array>, 'cnt': <int>, ...}, ...}
        index2word (dict):
            From word to an index - used for negative sampling
        vec_index2word (dict):
            Same as index2word but only words that have vectors
        unigram_table (dict):
            Negative sampling.
    '''
    def __init__(self):
        self.vocab = {}
        self.index2word = {}
        self.vec_index2word = {}
        self.unigram_table = []


    def inc_or_add(self, word, cnt=1, vec=None):
        r''' Add a word or incrase its count.

        Args:
            word (str):
                Word to be added
            cnt (str):
                By how much should the count be increased, or to wha
                should it be set if a new word.
            vec (np.array):
                Word vector
        '''
        if word not in self.vocab:
            self.add_word(word, cnt, vec)
        else:
            self.inc_wc(word, cnt)


    def remove_all_vectors(self):
        r''' Remove all stored vector representations
        '''
        self.vec_index2word = {}

        for word in self.vocab:
            self.vocab[word]['vec'] = None


    def remove_words_below_cnt(self, cnt):
        r''' Remove all words with frequency below cnt.

        Args:
            cnt (int):
                Word count limit.
        '''
        print("Words before removal: " + str(len(self.vocab)))
        for word in list(self.vocab.keys()):
            if self.vocab[word]['cnt'] < cnt:
                del self.vocab[word]
        print("Words after removal : " + str(len(self.vocab)))

        # Rebuild index2word and vec_index2word
        self.index2word = {}
        self.vec_index2word = {}
        for word in self.vocab.keys():
            ind = len(self.index2word)
            self.index2word[ind] = word
            self.vocab[word]['ind'] = ind

            if self.vocab[word]['vec'] is not None:
                self.vec_index2word[ind] = word


    def inc_wc(self, word, cnt=1):
        r''' Incraese word count by cnt.

        Args:
            word (str):
                For which word to increase the count
            cnt:
                By how muhc to incrase the count
        '''
        self.item(word)['cnt'] += cnt


    def add_vec(self, word, vec):
        r''' Add vector to a word.

        Args:
            word (str):
                To which word to add the vector.
            vec (np.array):
                The vector to add.
        '''
        self.vocab[word]['vec'] = vec

        ind = self.vocab[word]['ind']
        if ind not in self.vec_index2word:
            self.vec_index2word[ind] = word


    def reset_counts(self, cnt=1):
        r''' Reset the count for all word to cnt.

        Args:
            cnt (int):
                New count for all words in the vocab.
        '''
        for word in self.vocab.keys():
            self.vocab[word]['cnt'] = cnt

    def update_counts(self, tokens):
        r''' Given a list of tokens update counts for words in the vocab.

        Args:
            tokens (List[str]):
                Usually a large block of text split into tokens/words.
        '''
        for token in tokens:
            if token in self:
                self.inc_wc(token, 1)


    def add_word(self, word, cnt=1, vec=None, replace=True):
        """Add a word to the vocabulary

        Args:
            word (str):
                the word to be added, it should be lemmatized and lowercased
            cnt (int):
                count of this word in your dataset
            vec (np.array):
                the vector repesentation of the word
            replace (bool):
                will replace old vector representation
        """
        if word not in self.vocab:
            ind = len(self.index2word)
            self.index2word[ind] = word
            item = {'vec': vec, 'cnt': cnt, 'ind': ind}
            self.vocab[word] = item

            if vec is not None:
                self.vec_index2word[ind] = word
        elif replace and vec is not None:
            self.vocab[word]['vec'] = vec
            self.vocab[word]['cnt'] = cnt

            # If this word didn't have a vector before
            ind = self.vocab[word]['ind']
            if ind not in self.vec_index2word:
                self.vec_index2word[ind] = word


    def add_words(self, path, replace=True):
        """Adds words to the vocab from a file, the file
        is required to have the following format (vec being optional):
            <word>\t<cnt>[\t<vec_space_separated>]

        e.g. one line: the word house with 3 dimensional vectors
            house   34444   0.3232 0.123213 1.231231

        Args:
            path (str):
                path to the file with words and vectors
            replace (bool):
                existing words in the vocabulary will be replaced
        """
        f = open(path)

        for line in f:
            parts = line.split("\t")
            word = parts[0]
            cnt = int(parts[1].strip())
            vec = None
            if len(parts) == 3:
                vec = np.array([float(x) for x in parts[2].strip().split(" ")])

            self.add_word(word, cnt, vec)


    def make_unigram_table(self, table_size=100000000):
        r''' Make unigram table for negative sampling, look at the paper if interested
        in details.
        '''
        freqs = []
        self.unigram_table = []

        words = list(self.vec_index2word.values())
        for word in words:
            freqs.append(self[word])

        freqs = np.array(freqs)
        freqs = np.power(freqs, 3/4)
        sm = np.sum(freqs)

        for ind in self.vec_index2word.keys():
            word = self.vec_index2word[ind]
            f_ind = words.index(word)
            p = freqs[f_ind] / sm
            self.unigram_table.extend([ind] * int(p * table_size))

        self.unigram_table = np.array(self.unigram_table)


    def get_negative_samples(self, n=6, ignore_punct_and_num=False):
        r''' Get N negative samples.

        Args:
            n (int):
                How many words to return
            ignore_punct_and_num (bool):
                When returing words shold we skip punctuation and numbers.
        Returns:
            inds (List[int]):
                Indices for words in this vocabulary.
        '''
        if len(self.unigram_table) == 0:
            raise Exception("No unigram table present, please run the function vocab.make_unigram_table() first.")
        inds = np.random.randint(0, len(self.unigram_table), n)
        inds = self.unigram_table[inds]

        if ignore_punct_and_num:
            # Do not return anything that does not have letters in it
            inds = [ind for ind in inds if self.index2word[ind].upper().isupper()]

        return inds


    def __getitem__(self, word):
        return self.vocab[word]['cnt']


    def vec(self, word):
        return self.vocab[word]['vec']


    def item(self, word):
        return self.vocab[word]


    def __contains__(self, word):
        if word in self.vocab:
            return True

        return False


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)


    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            vocab = cls()
            vocab.__dict__ = pickle.load(f)
        return vocab
