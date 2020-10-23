import numpy as np
import pickle

class Vocab(object):
    def __init__(self):
        self.vocab = {}
        self.index2word = {}
        self.vec_index2word = {}
        self.unigram_table = []


    def inc_or_add(self, word, cnt=1, vec=None):
        if word not in self.vocab:
            self.add_word(word, cnt, vec)
        else:
            self.inc_wc(word)


    def remove_all_vectors(self):
        self.vec_index2word = {}

        for word in self.vocab:
            self.vocab[word]['vec'] = None


    def remove_words_below_cnt(self, cnt):
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


    def inc_wc(self, word):
        self.item(word)['cnt'] += 1


    def add_vec(self, word, vec):
        self.vocab[word]['vec'] = vec

        ind = self.vocab[word]['ind']
        if ind not in self.vec_index2word:
            self.vec_index2word[ind] = word


    def reset_counts(self):
        for word in self.vocab.keys():
            self.vocab[word]['cnt'] = 1

    def update_counts(self, tokens):
        for token in tokens:
            if token in self:
                self.vocab[token]['cnt'] += 1


    def add_word(self, word, cnt=1, vec=None, replace=True):
        """Add a word to the vocabulary

        word:  the word to be added, it should be lemmatized and lowercased
        cnt:  count of this word in your dataset
        vec:  the vector repesentation of the word
        replace:  will replace old vector representation
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

        path: path to the file with words and vectors
        replace:  existing words in the vocabulary will be replaced
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


    def make_unigram_table(self):
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
            self.unigram_table.extend([ind] * int(p * 100000000))

        self.unigram_table = np.array(self.unigram_table)


    def get_negative_samples(self, n=6, ignore_punct_and_num=False, stopwords=[]):
        if len(self.unigram_table) == 0:
            raise Exception("No unigram table present, please run the function vocab.make_unigram_table() first")
        inds = np.random.randint(0, len(self.unigram_table), n)
        inds = self.unigram_table[inds]

        if ignore_punct_and_num:
            # Do not return anything that does not have letters in it
            inds = [ind for ind in inds if (self.index2word[ind].upper().isupper() or "##" in self.index2word[ind])
                    and self.index2word[ind].lower() not in stopwords]

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
            pickle.dump(self, f)

    def save_dict(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)


    def load_dict(self, path):
        with open(path, 'rb') as f:
            self.__dict__ = pickle.load(f)


    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
