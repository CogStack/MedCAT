import numpy as np
import pickle

class Vocab(object):
    def __init__(self):
        self.vocab = {}
        self.index2word = []
        self.vec_index2word = []
        self.unigram_table = []


    def inc_or_add(self, word, cnt=1, vec=None):
        if word not in self.vocab:
            self.add_word(word, cnt, vec)
        else:
            self.inc_wc(word)


    def inc_wc(self, word):
        self.item(word)['cnt'] += 1


    def add_vec(self, word, vec):
        self.vocab[word]['vec'] = vec


    def add_word(self, word, cnt=1, vec=None):
        self.index2word.append(word)
        item = {'vec': None, 'cnt': cnt}
        self.vocab[word] = item


    def add_words(self, path):
        f = open(path)

        for line in f:
            parts = line.split("\t")
            self.index2word.append(parts[0])
            self.vec_index2word.append(parts[0])

            item = {'vec': np.array([float(x)
                for x in parts[2].strip().split(" ")]), 'cnt': int(parts[1].strip())}
            self.vocab[parts[0]] = item


    def add_words_nvec(self, path, reset_cnt=False):
        f = open(path)

        for line in f:
            parts = line.split("\t")
            if parts[0] not in self.vocab:
                self.index2word.append(parts[0])

                item = {'vec': None, 'cnt': int(parts[1].strip()),
                        'ind': len(self.index2word) - 1}
                self.vocab[parts[0]] = item
            elif reset_cnt:
                # Reset the count
                self.vocab[parts[0]]['cnt'] = int(parts[1].strip())


    def make_unigram_table(self):
        prob = []
        freqs = []

        for word in self.vec_index2word:
            freqs.append(self[word])

        freqs = np.array(freqs)
        freqs = np.power(freqs, 3/4)
        sm = np.sum(freqs)

        for i, word in enumerate(self.vec_index2word):
            p = freqs[i] / sm
            self.unigram_table.extend([i] * int(p * 1000000))

        self.unigram_table = np.array(self.unigram_table)


    def get_negative_samples(self, n=6):
        inds = np.random.randint(0, len(self.unigram_table), n)

        return self.unigram_table[inds]


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
