import os
import json
import pandas
import spacy
from time import sleep
from functools import partial
from multiprocessing import Process, Manager, Queue, Pool, Array
from medcat.cdb import CDB
from medcat.spacy_cat import SpacyCat
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.utils.spelling import CustomSpellChecker
from medcat.utils.spacy_pipe import SpacyPipe
from medcat.preprocessing.cleaners import spacy_tag_punct
from medcat.utils.helpers import get_all_from_name, tkn_inds_from_doc
from medcat.utils.loggers import basic_logger

log = basic_logger("CAT")

class CAT(object):
    """ Annotate a dataset
    """
    SEPARATOR = ""
    NESTED_ENTITIES = os.getenv("NESTED_ENTITIES", 'false').lower() == 'true'
    KEEP_PUNCT = os.getenv("KEEP_PUNCT", ":|.").split("|")

    def __init__(self, cdb, vocab=None, skip_stopwords=True, meta_cats=[]):
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

        for meta_cat in meta_cats:
            self.nlp.add_meta_cat(meta_cat, meta_cat.category_name)


    def __call__(self, text):
        return self.nlp(text)


    def add_concept_cntx(self, cui, text, tkn_inds, negative=False, lr=None, anneal=None, spacy_doc=None):
        if spacy_doc is None:
            spacy_doc = self(text)
        tkns = [spacy_doc[ind] for ind in range(tkn_inds[0], tkn_inds[-1] + 1)]
        self.spacy_cat._add_cntx_vec(cui=cui, doc=spacy_doc, tkns=tkns,
                                     negative=negative, lr=lr, anneal=anneal)


    def unlink_concept_name(self, cui, name, full_unlink=True):
        # Unlink a concept from a name
        o_name = str(name)
        p_name, tokens, _, _ = get_all_from_name(name=name, source_value=name, nlp=self.nlp)

        # To be sure unlink the orignal and the processed name
        names = [o_name, name, p_name, o_name.lower()]

        if tokens[-1].lower() == "s":
            # Remove last 's'
            names.append(p_name[0:-1])

        for name in names:
            cuis = [cui]
            if full_unlink and name in self.cdb.name2cui:
                cuis = list(self.cdb.name2cui[name])

            for cui in cuis:
                if cui in self.cdb.cui2names and name in self.cdb.cui2names[cui]:
                    self.cdb.cui2names[cui].remove(name)
                    if len(self.cdb.cui2names[cui]) == 0:
                        del self.cdb.cui2names[cui]

                if name in self.cdb.name2cui:
                    if cui in self.cdb.name2cui[name]:
                        self.cdb.name2cui[name].remove(cui)

                        if len(self.cdb.name2cui[name]) == 0:
                            del self.cdb.name2cui[name]



    def _add_name(self, cui, source_val, is_pref_name, only_new=False, desc=None, tui=None):
        onto = 'def'

        if cui in self.cdb.cui2ontos and self.cdb.cui2ontos[cui]:
            onto = list(self.cdb.cui2ontos[cui])[0]

        p_name, tokens, snames, tokens_vocab = get_all_from_name(name=source_val,
                source_value=source_val,
                nlp=self.nlp, version='clean')
        # This will add a new concept if the cui doesn't exist
        # or link the name to an existing concept if it exists.
        if cui not in self.cdb.cui2names or p_name not in self.cdb.cui2names[cui]:
            if not only_new or p_name not in self.cdb.name2cui:
                self.cdb.add_concept(cui, p_name, onto, tokens, snames, tokens_vocab=tokens_vocab,
                        original_name=source_val, is_pref_name=False, desc=desc, tui=tui)

        # Add the raw also if needed
        p_name, tokens, snames, tokens_vocab = get_all_from_name(name=source_val,
                source_value=source_val,
                nlp=self.nlp, version='raw')
        if cui not in self.cdb.cui2names or p_name not in self.cdb.cui2names[cui] or is_pref_name:
            if not only_new or p_name not in self.cdb.name2cui:
                self.cdb.add_concept(cui, p_name, onto, tokens, snames, tokens_vocab=tokens_vocab,
                                     original_name=source_val, is_pref_name=is_pref_name, desc=desc, tui=tui)

        # Fix for ntkns in cdb
        if p_name in self.cdb.name2ntkns:
            if len(tokens) not in self.cdb.name2ntkns[p_name]:
                self.cdb.name2ntkns[p_name].add(len(tokens))


    def add_name(self, cui, source_val, text=None, is_pref_name=False, tkn_inds=None, text_inds=None,
                 spacy_doc=None, lr=None, anneal=None, negative=False, only_new=False, desc=None, tui=None):
        """ Adds a new concept or appends the name to an existing concept
        if the cui already exists in the DB.

        cui:  Concept uniqe ID
        source_val:  Source value in the text
        text:  the text of a document where source_val was found
        """
        # First add the name
        self._add_name(cui, source_val, is_pref_name, only_new=only_new, desc=desc, tui=tui)

        # Now add context if text is present
        if (text is not None and (source_val in text or text_inds)) or \
           (spacy_doc is not None and (text_inds or tkn_inds)):
            if spacy_doc is None:
                spacy_doc = self(text)

            if tkn_inds is None:
                tkn_inds = tkn_inds_from_doc(spacy_doc=spacy_doc, text_inds=text_inds,
                                             source_val=source_val)

            if tkn_inds is not None and len(tkn_inds) > 0:
                self.add_concept_cntx(cui, text, tkn_inds, spacy_doc=spacy_doc, lr=lr, anneal=anneal,
                        negative=negative)


    def _print_stats(self, data, epoch=0, use_filters=False, use_overlaps=False):
        tp = 0
        fp = 0
        fn = 0
        docs_with_problems = set()
        if self.spacy_cat.TUI_FILTER is None:
            _tui_filter = None
        else:
            _tui_filter = list(self.spacy_cat.TUI_FILTER)
        if self.spacy_cat.CUI_FILTER is None:
            _cui_filter = None
        else:
            _cui_filter = list(self.spacy_cat.CUI_FILTER)

        # Stupid
        for project in data['projects']:
            cui_filter = None
            tui_filter = None

            if use_filters:
                if 'cuis' in project and len(project['cuis'].strip()) > 0:
                    cui_filter = [x.strip().upper() for x in project['cuis'].split(",")]
                if 'tuis' in project and len(project['tuis'].strip()) > 0:
                    tui_filter = [x.strip().upper() for x in project['tuis'].split(",")]

                self.spacy_cat.TUI_FILTER = tui_filter
                self.spacy_cat.CUI_FILTER = cui_filter

            for doc in project['documents']:
                spacy_doc = self(doc['text'])
                anns = doc['annotations']
                if use_overlaps:
                    p_anns = spacy_doc._.ents
                else:
                    p_anns = spacy_doc.ents

                anns_norm = []
                for ann in anns:
                    if ann.get('validated', True) and (not ann.get('killed', False) and not ann.get('deleted', False)):
                        anns_norm.append((ann['start'], ann['cui']))
                p_anns_norm = []
                for ann in p_anns:
                    p_anns_norm.append((ann.start_char, ann._.cui))

                for ann in p_anns_norm:
                    if ann in anns_norm:
                        tp += 1
                    else:
                        fp += 1
                        docs_with_problems.add(doc['name'])

                for ann in anns_norm:
                    if ann not in p_anns_norm:
                        fn += 1
                        docs_with_problems.add(doc['name'])

        try:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = (prec + rec) / 2
            print(tp, fp, fn)
            print("Epoch: {}, Prec: {}, Rec: {}, F1: {}".format(epoch, prec, rec, f1))
            print("First 10 out of {} docs with problems: {}".format(len(docs_with_problems),
                  "; ".join([str(x) for x in list(docs_with_problems)[0:10]])))
        except Exception as e:
            print(e)

        self.spacy_cat.TUI_FILTER = _tui_filter
        self.spacy_cat.CUI_FILTER = _cui_filter


    def train_supervised(self, data_path, reset_cdb=False, reset_cui_count=False, nepochs=10, lr=None,
                         anneal=None, print_stats=False, test_set=None, use_filters=False):
        """ Given data learns vector embeddings for concepts
        in a suppervised way.

        data_path:  path to data in json format
        """
        self.train = False
        data = json.load(open(data_path))

        if print_stats:
            if test_set:
                self._print_stats(test_set, use_filters=use_filters)
            else:
                self._print_stats(data, use_filters=use_filters)

        if reset_cdb:
            self.cdb = CDB()
            self.spacy_cat.cdb = self.cdb
            self.spacy_cat.cat_ann.cdb = self.cdb

        if reset_cui_count:
            # Get all CUIs
            cuis = []
            for project in data['projects']:
                for doc in project['documents']:
                    for ann in doc['annotations']:
                        cuis.append(ann['cui'])
            for cui in set(cuis):
                if cui in self.cdb.cui_count:
                    self.cdb.cui_count[cui] = 10

        # Remove entites that were terminated
        for project in data['projects']:
            for doc in project['documents']:
                for ann in doc['annotations']:
                    if ann.get('killed', False):
                        self.unlink_concept_name(ann['cui'], ann['value'])

        for epoch in range(nepochs):
            print("Starting epoch: {}".format(epoch))
            log.info("Starting epoch: {}".format(epoch))
            # Print acc before training

            for project in data['projects']:
                for i_doc, doc in enumerate(project['documents']):
                    spacy_doc = self(doc['text'])
                    for ann in doc['annotations']:
                        if not ann.get('killed', False):
                            cui = ann['cui']
                            start = ann['start']
                            end = ann['end']
                            deleted = ann.get('deleted', False)
                            self.add_name(cui=cui,
                                          source_val=ann['value'],
                                          spacy_doc=spacy_doc,
                                          text_inds=[start, end],
                                          negative=deleted,
                                          lr=lr,
                                          anneal=anneal)
            print("Done epoch, printing stats.")
            if print_stats:
                if test_set:
                    self._print_stats(test_set, epoch=epoch+1, use_filters=use_filters)
                else:
                    self._print_stats(data, epoch=epoch+1, use_filters=use_filters)



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


    def get_entities(self, text, cat_filter=None, only_cui=False):
        """ Get entities

        text:  text to be annotated
        return:  entities
        """
        doc = self(text)
        out = []

        if cat_filter:
            cat_filter(doc, self)

        out_ent = {}
        if self.NESTED_ENTITIES:
            _ents = doc._.ents
        else:
            _ents = doc.ents

        for ind, ent in enumerate(_ents):
            cui = str(ent._.cui)
            if not only_cui:
                out_ent['cui'] = cui
                out_ent['tui'] = str(ent._.tui)
                out_ent['type'] = str(self.cdb.tui2name.get(out_ent['tui'], ''))
                out_ent['source_value'] = str(ent.text)
                out_ent['acc'] = str(ent._.acc)
                out_ent['start'] = ent.start_char
                out_ent['end'] = ent.end_char
                out_ent['id'] = str(ent._.id)
                out_ent['pretty_name'] = self.cdb.cui2pretty_name.get(cui, '')

                if cui in self.cdb.cui2info and 'icd10' in self.cdb.cui2info[cui]:
                    icds = []
                    for icd10 in self.cdb.cui2info[cui]['icd10']:
                        icds.append(str(icd10['chapter']))
                    out_ent['icd10'] = ",".join(icds)
                else:
                    out_ent['icd10'] = ""

                if cui in self.cdb.cui2info and 'umls' in self.cdb.cui2info[cui]:
                    umls = [str(u) for u in self.cdb.cui2info[cui]['umls']]
                    out_ent['umls'] = ",".join(umls)
                else:
                    out_ent['umls'] = ''

                if cui in self.cdb.cui2info and 'snomed' in self.cdb.cui2info[cui]:
                    snomed = [str(u) for u in self.cdb.cui2info[cui]['snomed']]
                    out_ent['snomed'] = ",".join(snomed)
                else:
                    out_ent['snomed'] = ''

                if hasattr(ent._, 'meta_anns') and ent._.meta_anns:
                    out_ent['meta_anns'] = []
                    for key in ent._.meta_anns.keys():
                        one = {'name': key, 'value': ent._.meta_anns[key]}
                        out_ent['meta_anns'].append(one) 

                out.append(dict(out_ent))
            else:
                out.append(cui)

        return out


    def get_json(self, text, cat_filter=None, only_cui=False):
        """ Get output in json format

        text:  text to be annotated
        return:  json with fields {'entities': <>, 'text': text}
        """
        ents = self.get_entities(text, cat_filter, only_cui)
        out = {'entities': ents, 'text': text}

        return json.dumps(out)


    def multi_processing(self, in_data, nproc=8, batch_size=100, cat_filter=None, only_cui=False):
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
            p = Process(target=self._mp_cons, args=(in_q, out_dict, i, cat_filter, only_cui))
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
                out.extend(data)

        # Sometimes necessary to free memory
        out_dict.clear()
        del out_dict

        return out


    def _mp_cons(self, in_q, out_dict, pid=0, cat_filter=None, only_cui=False):
        cnt = 0
        out = []
        while True:
            if not in_q.empty():
                data = in_q.get()
                if data is None:
                    out_dict['pid: {}'.format(pid)] = out
                    break

                for id, text in data:
                    try:
                        doc = json.loads(self.get_json(text, cat_filter, only_cui))
                        out.append((id, doc))
                    except Exception as e:
                        print("Exception in _mp_cons")
                        print(e)

            sleep(1)
