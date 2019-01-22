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
from cat.preprocessing.iterators import EmbMimicCSV
from gensim.models import FastText
from multiprocessing import Process, Manager, Queue, Pool, Array
from time import sleep
import copy

class CAT(object):
    """ Annotate a dataset
    """
    def __init__(self, umls, vocab=None):
        self.umls = umls
        # Build the required spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all)
        self.nlp.add_punct_tagger(tagger=spacy_tag_punct)

        # Add spell checker pipe
        self.spell_checker = CustomSpellChecker(words=umls.vocab, big_vocab=vocab)
        self.nlp.add_spell_checker(spell_checker=self.spell_checker)

        # Add cat
        self.spacy_cat = SpacyCat(umls=umls, vocab=vocab)
        self.nlp.add_cat(spacy_cat=self.spacy_cat)


    def __call__(self, text):
        return self.nlp(text)


    def _mp_cons(self, in_q, out_dict, pid=0):
        cnt = 0
        while True:
            if not in_q.empty():
                data = in_q.get()
                if data is None:
                    out_dict['pid: {}'.format(pid)] = self.umls.get_train_dict()
                    break

                for text in data:
                    self.nlp(text)

            sleep(1)


    def multi_processing(self, data_iter, nproc=8, batch_size=4000):
        # Make a copy of umls training part
        _umls_train = copy.deepcopy(self.umls.get_train_dict())

        # Reset tr data, needed because of the final merge
        self.umls.cui_count = {}
        self.umls.cui2context_vec = {}
        self.umls.cui2context_words = {}
        self.umls.coo_dict = {}
        self.umls.cui2ncontext_vec

        in_q = Queue(maxsize=16)
        manager = Manager()
        out_dict = manager.dict()

        procs = []

        for i in range(nproc):
            p = Process(target=self._mp_cons, args=(in_q, out_dict, i))
            p.start()
            procs.append(p)

        cnt = 0
        data = []
        for text in data_iter:
            data.append(text)
            if len(data) == batch_size:
                in_q.put(data)
                data = []
                cnt += 1
                if cnt == 8:
                    break

        for _ in range(nproc):  # tell workers we're done
            in_q.put(None)

        for p in procs:
            p.join()

        # Get old data
        self.umls.merge_train_dict(_umls_train)

        # Merge all the new UMLS versions
        for key in out_dict.keys():
            data = out_dict[key]
            print("Merging training data for proc: " + str(key))
            self.umls.merge_train_dict(data)
            print(sum(self.umls.cui_count.values()))
            print(len(self.umls.cui_count))
