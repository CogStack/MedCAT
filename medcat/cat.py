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
from medcat.utils.data_utils import make_mc_train_test
import sys, traceback

log = basic_logger("CAT")

class CAT(object):
    r'''
    The main MedCAT class used to annotate documents, it is built on top of spaCy
    and works as a spaCy pipline. Creates an instance of a spaCy pipline that can
    be used as a spacy nlp model.

    Args:
        cdb (medcat.cdb.CDB):
            The concept database that will be used for NER+L
        vocab (medcat.utils.vocab.Vocab, optional):
            Vocabulary used for vector embeddings and spelling. Default: None
        skip_stopwords (bool, optional):
            If True the stopwords will be ignored and not detected in the pipeline.
            Default: True
        meta_cats (list of medcat.meta_cat.MetaCAT, optional):
            A list of models that will be applied sequentially on each
            detected annotation.

    Examples:
        >>>cat = CAT(cdb, vocab)
        >>>spacy_doc = cat("Put some text here")
        >>>print(spacy_doc.ents) # Detected entites
    '''
    def __init__(self, cdb, vocab=None, skip_stopwords=True, meta_cats=[], config={}):
        self.cdb = cdb
        self.vocab = vocab
        self.config = config

        # Build the spacy pipeline
        self.nlp = SpacyPipe(spacy_split_all)

        #self.nlp.add_punct_tagger(tagger=spacy_tag_punct)
        self.nlp.add_punct_tagger(tagger=partial(spacy_tag_punct,
                                                 skip_stopwords=skip_stopwords,
                                                 keep_punct=self.config.get("keep_punct", [':', '.'])))

        # Add spell checker
        self.spell_checker = CustomSpellChecker(cdb_vocab=self.cdb.vocab, data_vocab=self.vocab)
        self.nlp.add_spell_checker(spell_checker=self.spell_checker)

        # Add them cat class that does entity detection
        self.spacy_cat = SpacyCat(cdb=self.cdb, vocab=self.vocab)
        self.nlp.add_cat(spacy_cat=self.spacy_cat)

        # Add meta_annotaiton classes if they exist
        self._meta_annotations = False
        for meta_cat in meta_cats:
            self.nlp.add_meta_cat(meta_cat, meta_cat.category_name)
            self._meta_annotations = True


    def __call__(self, text):
        r'''
        Push the text through the pipeline.

        Args:
            text (string):
                The text to be annotated

        Returns:
            A spacy document with the extracted entities
        '''
        return self.nlp(text)


    def add_concept_cntx(self, cui, text, tkn_inds, negative=False, lr=None, anneal=None, spacy_doc=None):
        if spacy_doc is None:
            spacy_doc = self(text)
        tkns = [spacy_doc[ind] for ind in range(tkn_inds[0], tkn_inds[-1] + 1)]
        self.spacy_cat._add_cntx_vec(cui=cui, doc=spacy_doc, tkns=tkns,
                                     negative=negative, lr=lr, anneal=anneal)


    def unlink_concept_name(self, cui, name, full_unlink=True):
        names = [name, name.lower()]
        # Unlink a concept from a name
        p_name, tokens, _, _ = get_all_from_name(name=name, source_value=name, nlp=self.nlp, version='clean')
        # Add the clean version of the name
        names.append(p_name)
        # Get the raw version
        p_name, tokens, _, _ = get_all_from_name(name=name, source_value=name, nlp=self.nlp, version='raw')
        # Append the raw evrsion
        names.append(p_name)

        if tokens[-1].lower() == "s":
            # Remove last 's' - a stupid bug
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
        all_cuis = []

        if cui in self.cdb.cui2ontos and self.cdb.cui2ontos[cui]:
            onto = list(self.cdb.cui2ontos[cui])[0]

        # Add the original version of the name just lowercased
        p_name, tokens, snames, tokens_vocab = get_all_from_name(name=source_val,
                source_value=source_val,
                nlp=self.nlp, version='none')
        if cui not in self.cdb.cui2names or p_name not in self.cdb.cui2names[cui]:
            if not only_new or p_name not in self.cdb.name2cui:
                self.cdb.add_concept(cui, p_name, onto, tokens, snames, tokens_vocab=tokens_vocab,
                        original_name=source_val, is_pref_name=False, desc=desc, tui=tui)
        all_cuis.extend(self.cdb.name2cui[p_name])

        p_name, tokens, snames, tokens_vocab = get_all_from_name(name=source_val,
                source_value=source_val,
                nlp=self.nlp, version='clean')
        # This will add a new concept if the cui doesn't exist
        # or link the name to an existing concept if it exists.
        if cui not in self.cdb.cui2names or p_name not in self.cdb.cui2names[cui]:
            if not only_new or p_name not in self.cdb.name2cui:
                self.cdb.add_concept(cui, p_name, onto, tokens, snames, tokens_vocab=tokens_vocab,
                        original_name=source_val, is_pref_name=False, desc=desc, tui=tui)
        all_cuis.extend(self.cdb.name2cui[p_name])

        # Add the raw also if needed
        p_name, tokens, snames, tokens_vocab = get_all_from_name(name=source_val,
                source_value=source_val,
                nlp=self.nlp, version='raw')
        if cui not in self.cdb.cui2names or p_name not in self.cdb.cui2names[cui] or is_pref_name:
            if not only_new or p_name not in self.cdb.name2cui:
                self.cdb.add_concept(cui, p_name, onto, tokens, snames, tokens_vocab=tokens_vocab,
                                     original_name=source_val, is_pref_name=is_pref_name, desc=desc, tui=tui)
        all_cuis.extend(self.cdb.name2cui[p_name])

        # Fix for ntkns in cdb
        if p_name in self.cdb.name2ntkns:
            if len(tokens) not in self.cdb.name2ntkns[p_name]:
                self.cdb.name2ntkns[p_name].add(len(tokens))

        return list(set(all_cuis))


    def add_name(self, cui, source_val, text=None, is_pref_name=False, tkn_inds=None, text_inds=None,
                 spacy_doc=None, lr=None, anneal=None, negative=False, only_new=False, desc=None, tui=None,
                 manually_created=False):
        """ Adds a new concept or appends the name to an existing concept
        if the cui already exists in the DB.

        cui:  Concept uniqe ID
        source_val:  Source value in the text
        text:  the text of a document where source_val was found
        """
        # First add the name, get bac all cuis that link to this name
        all_cuis = self._add_name(cui, source_val, is_pref_name, only_new=only_new, desc=desc, tui=tui)

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

                if manually_created:
                    all_cuis.remove(cui)
                    for _cui in all_cuis:
                        self.add_concept_cntx(_cui, text, tkn_inds, spacy_doc=spacy_doc, lr=lr, anneal=anneal,
                                negative=True)


    def _print_stats(self, data, epoch=0, use_filters=False, use_overlaps=False, use_cui_doc_limit=False):
        tp = 0
        fp = 0
        fn = 0
        fps = {}
        fns = {}
        tps = {}
        cui_prec = {}
        cui_rec = {}
        cui_f1 = {}
        cui_counts = {}

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
                    cui_filter = [x.strip() for x in project['cuis'].split(",")]
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
                anns_norm_cui = []
                for ann in anns:
                    if (cui_filter is None and tui_filter is None) or (cui_filter is not None and ann['cui'] in cui_filter) or \
                       (tui_filter is not None and self.cdb.cui2tui.get(ann['cui'], 'unk') in tui_filter):
                        if ann.get('validated', True) and (not ann.get('killed', False) and not ann.get('deleted', False)):
                            anns_norm.append((ann['start'], ann['cui']))

                        if ann.get("validated", True):
                            # This is used to test was someone annotating for this CUI in this document
                            anns_norm_cui.append(ann['cui'])

                            if ann['cui'] in cui_counts:
                                cui_counts[ann['cui']] += 1
                            else:
                                cui_counts[ann['cui']] = 1

                p_anns_norm = []
                for ann in p_anns:
                    p_anns_norm.append((ann.start_char, ann._.cui))

                for ann in p_anns_norm:
                    if not use_cui_doc_limit or ann[1] in anns_norm_cui:
                        if ann in anns_norm:
                            tp += 1

                            if ann[1] in tps:
                                tps[ann[1]] += 1
                            else:
                                tps[ann[1]] = 1
                        else:
                            if ann[1] in fps:
                                fps[ann[1]] += 1
                            else:
                                fps[ann[1]] = 1
                            fp += 1
                        docs_with_problems.add(doc['name'])

                for ann in anns_norm:
                    if ann not in p_anns_norm:
                        fn += 1
                        docs_with_problems.add(doc['name'])

                        if ann[1] in fns:
                            fns[ann[1]] += 1
                        else:
                            fns[ann[1]] = 1
        try:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = (prec + rec) / 2
            print("Epoch: {}, Prec: {}, Rec: {}, F1: {}".format(epoch, prec, rec, f1))
            print("First 10 out of {} docs with problems: {}".format(len(docs_with_problems),
                  "; ".join([str(x) for x in list(docs_with_problems)[0:10]])))

            # Sort fns & prec
            fps = {k: v for k, v in sorted(fps.items(), key=lambda item: item[1], reverse=True)}
            fns = {k: v for k, v in sorted(fns.items(), key=lambda item: item[1], reverse=True)}
            tps = {k: v for k, v in sorted(tps.items(), key=lambda item: item[1], reverse=True)}


            # F1 per concept
            for cui in tps.keys():
                prec = tps[cui] / (tps.get(cui, 0) + fps.get(cui, 0))
                rec = tps[cui] / (tps.get(cui, 0) + fns.get(cui, 0))
                f1 = (prec + rec) / 2
                cui_prec[cui] = prec
                cui_rec[cui] = rec
                cui_f1[cui] = f1


            # Get top 10
            pr_fps = [(self.cdb.cui2pretty_name.get(cui,
                list(self.cdb.cui2original_names.get(cui, ["UNK"]))[0]), cui, fps[cui]) for cui in list(fps.keys())[0:10]]
            pr_fns = [(self.cdb.cui2pretty_name.get(cui,
                list(self.cdb.cui2original_names.get(cui, ["UNK"]))[0]), cui, fns[cui]) for cui in list(fns.keys())[0:10]]
            pr_tps = [(self.cdb.cui2pretty_name.get(cui,
                list(self.cdb.cui2original_names.get(cui, ["UNK"]))[0]), cui, tps[cui]) for cui in list(tps.keys())[0:10]]


            print("\n\nFalse Positives\n")
            for one in pr_fps:
                print("{:70} - {:20} - {:10}".format(one[0], one[1], one[2]))
            print("\n\nFalse Negatives\n")
            for one in pr_fns:
                print("{:70} - {:20} - {:10}".format(one[0], one[1], one[2]))
            print("\n\nTrue Positives\n")
            for one in pr_tps:
                print("{:70} - {:20} - {:10}".format(one[0], one[1], one[2]))
            print("*"*110 + "\n")


        except Exception as e:
            traceback.print_exc()

        self.spacy_cat.TUI_FILTER = _tui_filter
        self.spacy_cat.CUI_FILTER = _cui_filter

        return fps, fns, tps, cui_prec, cui_rec, cui_f1, cui_counts


    def train_supervised(self, data_path, reset_cdb=False, reset_cui_count=False, nepochs=30, lr=None,
                         anneal=None, print_stats=True, use_filters=False, terminate_last=False, use_cui_doc_limit=False,
                         test_size=0, seed=17):
        """ Given data learns vector embeddings for concepts
        in a suppervised way.

        data_path:  path to data in json format
        """
        self.train = False
        data = json.load(open(data_path))
        cui_counts = {}

        if test_size == 0:
            test_set = data
            train_set = data
        else:
            train_set, test_set, _, _ = make_mc_train_test(data, self.cdb, seed=seed, test_size=test_size)

            # Add all names, as we could have some in test set
            for project in data['projects']:
                for document in project['documents']:
                    for ann in document['annotations']:
                        if not ann.get('killed', False):
                            self.add_name(ann['cui'], ann['value'])

        if print_stats:
            self._print_stats(test_set, use_filters=use_filters, use_cui_doc_limit=use_cui_doc_limit)


        if reset_cdb:
            self.cdb = CDB()
            self.spacy_cat.cdb = self.cdb
            self.spacy_cat.cat_ann.cdb = self.cdb

        if reset_cui_count:
            # Get all CUIs
            cuis = []
            for project in train_set['projects']:
                for doc in project['documents']:
                    for ann in doc['annotations']:
                        cuis.append(ann['cui'])
            for cui in set(cuis):
                if cui in self.cdb.cui_count:
                    self.cdb.cui_count[cui] = 10

        # Remove entites that were terminated
        for project in train_set['projects']:
            for doc in project['documents']:
                for ann in doc['annotations']:
                    if ann.get('killed', False):
                        self.unlink_concept_name(ann['cui'], ann['value'])

        for epoch in range(nepochs):
            print("Starting epoch: {}".format(epoch))
            log.info("Starting epoch: {}".format(epoch))
            # Print acc before training

            for project in train_set['projects']:
                for i_doc, doc in enumerate(project['documents']):
                    spacy_doc = self(doc['text'])
                    for ann in doc['annotations']:
                        if not ann.get('killed', False):
                            cui = ann['cui']
                            start = ann['start']
                            end = ann['end']
                            deleted = ann.get('deleted', False)
                            manually_created = False
                            if ann.get('manually_created', False) or ann.get('alternative', False):
                                manually_created = True
                            self.add_name(cui=cui,
                                          source_val=ann['value'],
                                          spacy_doc=spacy_doc,
                                          text_inds=[start, end],
                                          negative=deleted,
                                          lr=lr,
                                          anneal=anneal,
                                          manually_created=manually_created)
            if terminate_last:
                # Remove entites that were terminated, but after all training is done
                for project in train_set['projects']:
                    for doc in project['documents']:
                        for ann in doc['annotations']:
                            if ann.get('killed', False):
                                self.unlink_concept_name(ann['cui'], ann['value'])

            if epoch % 5 == 0:
                if print_stats:
                    fp, fn, tp, p, r, f1, cui_counts = self._print_stats(test_set, epoch=epoch+1,
                                                             use_filters=use_filters,
                                                             use_cui_doc_limit=use_cui_doc_limit)
        return fp, fn, tp, p, r, f1, cui_counts

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
        if self.config.get('nested_entities', False):
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

        if self._meta_annotations:
            # Hack for torch using multithreading, which is not good here
            import torch
            torch.set_num_threads(1)

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
