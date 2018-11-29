""" Representation class for UMLS data
"""
import pickle

class UMLS(object):
    """ Holds all the UMLS data required for annotation

    stopwords:  Words to skip for cui vocab
    """
    def __init__(self, stopwords=[]):
        self.stopwords = stopwords
        self.name2cui = {}
        self.cui2names = {}
        self.sname2name = set()
        self.cui2words = {}
        self.onto2cuis = {}
        self.cui2context_vec = {}
        self.cui2context_words = {}
        self.cui2doc_words = {}
        self.vocab = {}
        self.cui2avg_weight = {}

        # Think about this for big sets and vocabs
        #self.stringstore = StringStore()

        self.CONTEXT_WORDS_LIMIT = 200

    def add_concept(self, cui, name, onto, tokens, snames):
        """ Add a concept to internal UMLS representation

        cui:  Identifier
        name:  Concept name
        onto:  Ontology from which the concept is taken
        tokens:  A list of words existing in the name
        snames:  if name is "heart attack" snames is
                 ['heart', 'heart attack']
        """
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
            self.cui2avg_weight[cui] = 1
        for token in tokens:
            if token not in self.stopwords and not token.isdigit() and len(token) > 1:
                if token in self.cui2words[cui]:
                    self.cui2words[cui][token] += 1
                else:
                    self.cui2words[cui][token] = 1



    def add_context_vec(self, cui, context_vec):
        """ Add the vector representation of a context for this CUI

        cui:  The concept in question
        context_vec:  Vector represenation of the context
        """
        if cui in self.cui2context_vec:
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


    def add_doc_words(self, cui, doc_words):
        """ Add words that appear in the document of this CUI

        cui:  The concept in question
        doc_words:  Array of words that appeard in the document
        """

        # Clean up context words, remove if not in vocab and if stopword
        doc_words = [x for x in doc_words if len(x) > 1 and
                     x not in self.stopwords and x in self.vocab and not x.isdigit()]

        if cui in self.cui2doc_words:
            vcb = self.cui2doc_words[cui]

            for word in doc_words:
                if word in vcb:
                    vcb[word] += 1
                else:
                    vcb[word] = 1
        else:
            self.cui2doc_words[cui] = {}
            vcb = self.cui2doc_words[cui]

            for word in doc_words:
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

