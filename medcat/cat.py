import pandas
import spacy
from spacy.tokenizer import Tokenizer
from medcat.cdb import CDB
from medcat.spacy_cat import SpacyCat
from medcat.preprocessing.tokenizers import spacy_split_all
from spacy.tokens import Token
from medcat.utils.spelling import CustomSpellChecker, SpacySpellChecker
from medcat.utils.spacy_pipe import SpacyPipe
from medcat.preprocessing.iterators import EmbMimicCSV
from gensim.models import FastText
from multiprocessing import Process, Manager, Queue, Pool, Array
from time import sleep
import copy
import json
from functools import partial
from medcat.preprocessing.cleaners import spacy_tag_punct
from medcat.utils.helpers import get_all_from_name, tkn_inds_from_doc
import os
from medcat.utils.loggers import basic_logger

log = basic_logger("CAT")

class CAT(object):
    """ Annotate a dataset
    """
    SEPARATOR = ""
    NESTED_ENTITIES = os.getenv("NESTED_ENTITIES", 'false').lower() == 'true'
    KEEP_PUNCT = os.getenv("KEEP_PUNCT", ":").split("|")

    def __init__(self, cdb, vocab=None, skip_stopwords=True):
        self.cdb = cdb
        self.vocab = vocab
        # Build the required spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all)
        #self.nlp.add_punct_tagger(tagger=spacy_tag_punct)
        self.nlp.add_punct_tagger(tagger=partial(spacy_tag_punct,
                                                 skip_stopwords=skip_stopwords,
                                                 keep_punct=self.KEEP_PUNCT))

        # Add spell checker pipe
        self.spell_checker = CustomSpellChecker(cdb_vocab=self.cdb.vocab, data_vocab=self.vocab)
        self.nlp.add_spell_checker(spell_checker=self.spell_checker)

        # Add cat
        self.spacy_cat = SpacyCat(cdb=self.cdb, vocab=self.vocab)
        self.nlp.add_cat(spacy_cat=self.spacy_cat)


    def __call__(self, text):
        return self.nlp(text)


    def add_concept_cntx(self, cui, text, tkn_inds, negative=False, lr=0.1, anneal=False, spacy_doc=None):
        if spacy_doc is None:
            spacy_doc = self(text)
        tkns = [spacy_doc[ind] for ind in range(tkn_inds[0], tkn_inds[-1] + 1)]
        self.spacy_cat._add_cntx_vec(cui=cui, doc=spacy_doc, tkns=tkns,
                                     negative=negative, lr=lr, anneal=anneal)


    def unlink_concept_name(self, cui, name):
        # Unlink a concept from a name
        p_name, _, _, _ = get_all_from_name(name=name, source_value=name, nlp=self.nlp)

        # To be sure unlink the orignal and the processed name
        names = [name, p_name]
        for name in names:
            if name in self.cdb.cui2names[cui]:
                self.cdb.cui2names[cui].remove(name)
                if len(self.cdb.cui2names[cui]) == 0:
                    del self.cdb.cui2names[cui]

            if name in self.cdb.name2cui:
                if cui in self.cdb.name2cui[name]:
                    self.cdb.name2cui[name].remove(cui)

                    if len(self.cdb.name2cui[name]) == 0:
                        del self.cdb.name2cui[name]


    def _add_name(self, cui, source_val, is_pref_name):
        onto = 'def'
        if cui in self.cdb.cui2ontos:
            onto = self.cdb.cui2ontos[cui][0]
        p_name, tokens, snames, tokens_vocab = get_all_from_name(name=source_val, source_value=source_val,
                nlp=self.nlp, version='clean')

        # This will add a new concept if the cui doesn't exist
        #or link the name to an existing concept if it exists.
        self.cdb.add_concept(cui, p_name, onto, tokens, snames, tokens_vocab=tokens_vocab,
                original_name=source_val, is_pref_name=is_pref_name)

        # Add the raw also
        p_name, tokens, snames, tokens_vocab = get_all_from_name(name=source_val, source_value=source_val,
                nlp=self.nlp, version='raw')
        self.cdb.add_concept(cui, p_name, onto, tokens, snames, tokens_vocab=tokens_vocab,
                original_name=source_val, is_pref_name=False)


    def add_name(self, cui, source_val, text=None, is_pref_name=False, tkn_inds=None, text_inds=None, spacy_doc=None, lr=0.1, anneal=False):
        """ Adds a new concept or appends the name to an existing concept
        if the cui already exists in the DB.

        cui:  Concept uniqe ID
        source_val:  Source value in the text
        text:  the text of a document where source_val was found
        """

        # First add the name
        self._add_name(cui, source_val, is_pref_name)

        # Now add context if text is present
        if text is not None and (source_val in text or text_inds):
            if spacy_doc is None:
                spacy_doc = self(text)

            if tkn_inds is None:
                tkn_inds = tkn_inds_from_doc(spacy_doc=spacy_doc, text_inds=text_inds,
                                             source_val=source_val)

            if tkn_inds is not None and len(tkn_inds) > 0:
                self.add_concept_cntx(cui, text, tkn_inds, spacy_doc=spacy_doc, lr=lr, anneal=anneal)


    def train_supervised(self, data):
        """ Given data learns vector embeddings for concepts
        in a suppervised way.

        data:  json data in format <>
        """
        pass

    @property
    def train(self):
        return self.spacy_cat.train


    @train.setter
    def train(self, val):
        self.spacy_cat.train = val


    def run_training(self, data_iterator, fine_tune=False):
        """ Runs training on the data

        data_iterator:  Simple iterator over sentences/documents, e.g. a open file
                         or an array or anything else that we can use in a for loop.
        fine_tune:  If False old training will be removed
        """
        self.train = True
        cnt = 0

        if not fine_tune:
            print("Removing old training data!\n")
            self.cdb.reset_training()
            self.cdb.coo_dict = {}
            self.spacy_cat._train_skip_names = {}

        for line in data_iterator:
            if line is not None:
                try:
                    _ = self(line)
                except Exception as e:
                    print("LINE: '{}' \t WAS SKIPPED".format(line))
                    print("BECAUSE OF: " + str(e))
                if cnt % 1000 == 0:
                    print("DONE: " + str(cnt))
                cnt += 1
        self.train = False


    def get_entities(self, text):
        """ Get entities

        text:  text to be annotated
        return:  entities
        """
        doc = self(text)
        out = []

        out_ent = {}
        if self.NESTED_ENTITIES:
            _ents = doc._.ents
        else:
            _ents = doc.ents

        for ind, ent in enumerate(_ents):
            out_ent['cui'] = str(ent._.cui)
            out_ent['tui'] = str(ent._.tui)
            out_ent['type'] = str(self.cdb.tui2name.get(out_ent['tui'], ''))
            out_ent['source_value'] = str(ent.text)
            out_ent['acc'] = str(ent._.acc)
            out_ent['start_tkn'] = ent[0].i
            out_ent['end_tkn'] = ent[-1].i
            out_ent['start_ind'] = ent.start_char
            out_ent['end_ind'] = ent.end_char
            out_ent['label'] = str(ent.label_)
            out_ent['id'] = str(ent._.id)
            out_ent['pretty_name'] = self.cdb.cui2pretty_name.get(ent._.cui, '')
            out.append(dict(out_ent))

        return out


    def get_json(self, text):
        """ Get output in json format

        text:  text to be annotated
        return:  json with fields {'entities': <>, 'text': text}
        """
        ents = self.get_entities(text)
        out = {'entities': ents, 'text': text}

        return json.dumps(out)

    def multi_processing(self, in_data, nproc=8, batch_size=100):
        """ Run multiprocessing NOT FOR TRAINING
        in_data:  an iterator or array with format: [(id, text), (id, text), ...]
        nproc:  number of processors
        batch_size:  obvious

        return:  an list of tuples: [(id, doc_json), (id, doc_json), ...]
        """

        # Create the input output for MP
        in_q = Queue(maxsize=4*nproc)
        manager = Manager()
        out_dict = manager.dict()
        out_dict['processed'] = []

        # Create processes
        procs = []
        for i in range(nproc):
            p = Process(target=self._mp_cons, args=(in_q, out_dict, i))
            p.start()
            procs.append(p)

        data = []
        for id, text in in_data:
            data.append((id, text))
            if len(data) == batch_size:
                in_q.put(data)
                data = []
        # Put the last batch if it exists
        if len(data) > 0:
            in_q.put(data)

        for _ in range(nproc):  # tell workers we're done
            in_q.put(None)

        for p in procs:
            p.join()

        # Close the queue as it can cause memory leaks
        in_q.close()

        out = []
        for key in out_dict.keys():
            if 'pid' in key:
                data = out_dict[key]
                print("Merging training data for proc: " + str(key))
                out.extend(data[2])
        print("Done processing {} documents\n".format(len(out)))
        return out


    def multi_processing_coo(self, in_data, nproc=8, batch_size=100, coo=False):
        """ Run multiprocessing NOT FOR TRAINING
        in_data:  an iterator or array with format: [(id, text), (id, text), ...]
        nproc:  number of processors

        return:  an list of tuples: [(id, doc_json), (id, doc_json), ...]
        """

        # Create the input output for MP
        in_q = Queue(maxsize=4*nproc)
        manager = Manager()
        out_dict = manager.dict()
        out_dict['processed'] = []

        # Create processes
        procs = []
        for i in range(nproc):
            p = Process(target=self._mp_cons, args=(in_q, out_dict, i))
            p.start()
            procs.append(p)

        data = []
        for id, text in in_data:
            data.append((id, text))
            if len(data) == batch_size:
                in_q.put(data)
                data = []
        # Put the last batch if it exists
        if len(data) > 0:
            in_q.put(data)

        for _ in range(nproc):  # tell workers we're done
            in_q.put(None)

        in_q.close()

        for p in procs:
            p.join()

        # Merge all the new CDB versions and get the output
        out = []
        for key in out_dict.keys():
            if 'pid' in key:
                data = out_dict[key]
                print("Merging training data for proc: " + str(key))
                out.extend(data[2])
        return out


    def _mp_cons(self, in_q, out_dict, pid=0):
        cnt = 0
        out = []
        while True:
            if not in_q.empty():
                data = in_q.get()
                if data is None:
                    print("DONE " + str(pid))
                    out_dict['pid: {}'.format(pid)] = (self.cdb.coo_dict,
                            self.cdb.cui_count_ext, out)
                    break

                for id, text in data:
                    try:
                        doc = json.loads(self.get_json(text))
                        out.append((id, doc))
                    except Exception as e:
                        print(e)

            sleep(1)


