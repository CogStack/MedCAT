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
from medcat.utils.helpers import get_all_from_name

class CAT(object):
    """ Annotate a dataset
    """
    SEPARATOR = ""
    def __init__(self, cdb, vocab=None, skip_stopwords=True):
        self.cdb = cdb
        # Build the required spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all)
        #self.nlp.add_punct_tagger(tagger=spacy_tag_punct)
        self.nlp.add_punct_tagger(tagger=partial(spacy_tag_punct, skip_stopwords=skip_stopwords))

        # Add spell checker pipe
        self.spell_checker = CustomSpellChecker(cdb_vocab=cdb.vocab, data_vocab=vocab)
        self.nlp.add_spell_checker(spell_checker=self.spell_checker)

        # Add cat
        self.spacy_cat = SpacyCat(cdb=cdb, vocab=vocab)
        self.nlp.add_cat(spacy_cat=self.spacy_cat)


    def __call__(self, text):
        return self.nlp(text)


    def add_concept_cntx(self, cui, text, tkn_inds, negative=False):
        doc = self(text)
        tkns = [doc[ind] for ind in range(tkn_inds[0], tkn_inds[-1] + 1)]
        self.spacy_cat._add_cntx_vec(cui=cui, doc=doc, tkns=tkns, manual=True,
                                     negative=negative)


    def add_concept(self, concept, text=None, tkn_inds=None):
        cui = concept['cui']
        onto = concept.get('onto', 'user')
        pretty_name = concept['name']
        source_value = concept['source_value']
        name, tokens, snames, tokens_vocab = get_all_from_name(name=pretty_name, source_value=source_value, nlp=self.nlp)
        tui = concept.get('tui', 'None')
        unique = True

        print(cui, name, onto, tokens, snames, tokens_vocab, tui)

        # Add the new concept
        self.cdb.add_concept(cui, name, onto, tokens, snames,
                             tui=tui, pretty_name=pretty_name, is_pref_name=True,
                             tokens_vocab=tokens_vocab, unique=unique)

        if tkn_inds and text:
            # Add the context
            self.add_concept_cntx(cui, text, tkn_inds)


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
            print("Removing old training data, please make sure this is what you want")
            self.cdb.reset_training()
            self.cdb.coo_dict = {}
            self.spacy_cat._train_skip_names = {}

        for line in data_iterator:
            if line is not None:
                try:
                    _ = self(line)
                except Exception as e:
                    # TODO: make nice
                    print(e)
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
        #TODO: should we use .ents or ._.ents
        for ind, ent in enumerate(doc._.ents):
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


    def multi_processing(self, in_data, nproc=8, batch_size=100, coo=False):
        """ Run multiprocessing NOT FOR TRAINING
        in_data:  an iterator or array with format: [(id, text), (id, text), ...]
        nproc:  number of processors

        return:  an list of tuples: [(id, doc_json), (id, doc_json), ...]
        """

        # TODO: reorganize a abit, quite a mess here

        # Make a copy of cdb training part
        cui_count_ext = copy.deepcopy(self.cdb.cui_count_ext)
        coo_dict = copy.deepcopy(self.cdb.coo_dict)

        # Reset the cui_count_ext and coo_dict
        self.cdb.cui_count_ext = {}
        self.cdb.coo_dict = {}

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

        # Add the saved counts
        self.cdb.merge_run_only(coo_dict=coo_dict, cui_count_ext=cui_count_ext)
        # Merge all the new CDB versions and get the output
        out = []
        for key in out_dict.keys():
            if 'pid' in key:
                data = out_dict[key]
                print("Merging training data for proc: " + str(key))
                print(sum(self.cdb.cui_count_ext.values()))
                self.cdb.merge_run_only(coo_dict=data[0], cui_count_ext=data[1])
                print(sum(self.cdb.cui_count_ext.values()))
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


