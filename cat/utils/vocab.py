class Vocab(object):
    def __init__(self, size=300):
        self.size = size
        self.items = {}

    def add_from_vocab(self, vocab):
        vec = np.zeros(self.size)
        for key in vocab.keys():
            self.items[key] = Item(


class Item(object):
    def __init__(self):
        self.vec = vec
        self.count = count
