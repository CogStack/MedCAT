""" Representation class for UMLS data
"""
import pickle
import numpy as np
from scipy.sparse import dok_matrix
from gensim.matutils import unitvec

class UMLS(object):
    """ Holds all the UMLS data required for annotation

    stopwords:  Words to skip for cui vocab
    """
    def __init__(self, stopwords=[]):
        self.index2cui = []
        self.cui2index = {}
        self.stopwords = stopwords
        self.name2cui = {}
        self.name2cnt = {}
        self.name_isupper = {}
        self.cui_count = {}
        self.cui2names = {}
        self.cui2tui = {}
        self.cui2pref_name = {}
        self.sname2name = set()
        self.cui2words = {}
        self.onto2cuis = {}
        self.cui2context_vec = {}
        self.cui2ncontext_vec = {}
        self.cui2context_words = {}
        self.vocab = {}
        self.cui2scores = {}
        self._coo_matrix = None
        self.coo_dict = {}

        # Think about this for big sets and vocabs
        #self.stringstore = StringStore()

        self.CONTEXT_WORDS_LIMIT = 80

    def add_concept(self, cui, name, onto, tokens, snames, isupper, is_pref_name=False, tui=None):
        """ Add a concept to internal UMLS representation

        cui:  Identifier
        name:  Concept name
        onto:  Ontology from which the concept is taken
        tokens:  A list of words existing in the name
        snames:  if name is "heart attack" snames is
                 ['heart', 'heart attack']
        isupper:  If name in the original ontology is upper_cased
        is_pref_name:  If this is the prefered name for this CUI
        tui:  Semantic type
        """
        # Add is name upper
        if name in self.name_isupper:
            self.name_isupper[name] = self.name_isupper[name] or isupper
        else:
            self.name_isupper[name] = isupper

        if is_pref_name:
            self.cui2pref_name[cui] = name

        if tui is not None:
            self.cui2tui[cui] = tui

        # Add name to cnt
        if name not in self.name2cnt:
            self.name2cnt[name] = {}
        if cui in self.name2cnt[name]:
            self.name2cnt[name][cui] += 1
        else:
            self.name2cnt[name][cui] = 1

        # Add cui to a list of cuis
        if cui not in self.index2cui:
            self.index2cui.append(cui)
            self.cui2index[cui] = len(self.index2cui) - 1

            # Expand coo matrix if it is used
            if self._coo_matrix is not None:
                s = self._coo_matrix.shape[0] + 1
                self._coo_matrix.resize((s, s))

        # Add words to vocab
        for token in tokens:
            if token in self.vocab:
                self.vocab[token] += 1
            else:
                self.vocab[token] = 1

        # Add mappings to onto2cuis
        if onto not in self.onto2cuis:
            self.onto2cuis[onto] = set([cui])
        else:
            self.onto2cuis[onto].add(cui)

        # Add mappings to name2cui
        if name not in self.name2cui:
            self.name2cui[name] = set([cui])
        else:
            self.name2cui[name].add(cui)

        # Add snames to set
        self.sname2name.update(snames)

        # Add mappings to cui2names
        if cui not in self.cui2names:
            self.cui2names[cui] = set([name])
        else:
            self.cui2names[cui].add(name)

        # Add mappings to cui2words
        if cui not in self.cui2words:
            self.cui2words[cui] = {}
        for token in tokens:
            if token not in self.stopwords and not token.isdigit() and len(token) > 1:
                if token in self.cui2words[cui]:
                    self.cui2words[cui][token] += 1
                else:
                    self.cui2words[cui][token] = 1


    def _add_scores(self, cui, v1, v2):
        vec_sim = np.dot(unitvec(v1), unitvec(v2))

        if cui not in self.cui2scores:
            self.cui2scores[cui] = {}

        if 'vec' not in self.cui2scores[cui]:
            self.cui2scores[cui]['vec'] = {}
            self.cui2scores[cui]['vec']['min'] = 1
            self.cui2scores[cui]['vec']['max'] = 0
            self.cui2scores[cui]['vec']['avg'] = 1
        else:
            _min = self.cui2scores[cui]['vec']['min']
            _max = self.cui2scores[cui]['vec']['max']
            _avg = self.cui2scores[cui]['vec']['avg']

            if vec_sim < _min:
                self.cui2scores[cui]['vec']['min'] = vec_sim
            if vec_sim > _max:
                self.cui2scores[cui]['vec']['max'] = vec_sim

            # Add average
            self.cui2scores[cui]['vec']['avg'] = (_avg + vec_sim) / 2


    def add_context_vec(self, cui, context_vec, negative=False):
        """ Add the vector representation of a context for this CUI

        cui:  The concept in question
        context_vec:  Vector represenation of the context
        """

        # TODO: Missing negative context, try noise contrasite estimate, even though
        #this already works really nicely - without a test set hard to tell is negative needed
        sim = 0
        cv = context_vec
        if cui in self.cui2context_vec:
            sim = np.dot(unitvec(cv), unitvec(self.cui2context_vec[cui]))

            if negative:
                b = min((1 / self.cui_count[cui]), 0.1)  * max(0, sim)
                self.cui2context_vec[cui] = self.cui2context_vec[cui]*(1-b) - cv*b
            else:
                c = 0.1
                if sim < 0.2:
                    b = max((1 / self.cui_count[cui]), c)  * (1 - max(0, sim))
                    self.cui2context_vec[cui] = self.cui2context_vec[cui]*(1-b) + cv*b
        else:
            self.cui2context_vec[cui] = cv

        """
        # Second option, once a test set is there - try booth
        sim = 0
        if cui in self.cui2context_vec:
            sim = max(0, np.dot(unitvec(context_vec), unitvec(self.cui2context_vec[cui])))
        if sim < 0.2:
            if cui in self.cui2context_vec:
                self._add_scores(cui, context_vec, self.cui2context_vec[cui])
                self.cui2context_vec[cui] = self.cui2context_vec[cui]*0.95 + context_vec*0.05
            else:
                self._add_scores(cui, context_vec, context_vec)
                self.cui2context_vec[cui] = context_vec
            #print(self.cui2context_vec[cui])
        """

    def add_ncontext_vec(self, cui, ncontext_vec):
        """ Add the vector representation of a context for this CUI

        cui:  The concept in question
        ncontext_vec:  Vector represenation of the context
        """
        if cui in self.cui2ncontext_vec:
            self.cui2context_vec[cui] = (self.cui2context_vec[cui] + context_vec) / 2
        else:
            self.cui2context_vec[cui] = context_vec


    def add_context_words(self, cui, context_words):
        """ Add words that appear in the context of this CUI

        cui:  The concept in question
        context_words:  Array of words that appeard in the context
        """
        if cui in self.cui2context_words:
            vcb = self.cui2context_words[cui]

            for word in context_words:
                if word in vcb:
                    vcb[word] += 1
                else:
                    vcb[word] = 1
        else:
            self.cui2context_words[cui] = {}
            vcb = self.cui2context_words[cui]

            for word in context_words:
                if word in vcb:
                    vcb[word] += 1
                else:
                    vcb[word] = 1

        if len(vcb) > self.CONTEXT_WORDS_LIMIT:
            # Remove 1/3 of the words with lowest frequency
            remove_from = int(self.CONTEXT_WORDS_LIMIT / 3 * 2)
            keys = [k for k in sorted(vcb, key=vcb.get, reverse=True)][remove_from:]
            for key in keys:
                del vcb[key]


    def add_coo(self, cui1, cui2):
        key = [self.cui2index[cui1], self.cui2index[cui2]]
        key.sort()
        key = tuple(key)

        if key in self.coo_dict:
            self.coo_dict[key] += 1
        else:
            self.coo_dict[key] = 1

    def add_coos(self, cuis):
        cnt = 0
        for i, cui1 in enumerate(cuis):
            for cui2 in cuis[i:]:
                cnt += 1
                self.add_coo(cui1, cui2)

    @property
    def coo_matrix(self):
        if self._coo_matrix is None:
            s = len(self.cui2index)
            self._coo_matrix = dok_matrix((s, s), dtype=np.uint32)

        self._coo_matrix._update(self.coo_dict)
        return self._coo_matrix

    @coo_matrix.setter
    def coo_matrix(self, val):
        raise AttributeError("Can not set attribute coo_matrix")


    def merge(self, umls):
        """ Merges another umls instance into this one

        umls:  To be merged with this one
        """
        pass


    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

