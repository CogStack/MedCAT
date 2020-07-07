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
import time
import sys, traceback
from tqdm.autonotebook import tqdm

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
        skip_stopwords (bool):
            If True the stopwords will be ignored and not detected in the pipeline.
            Default: True
        meta_cats (list of medcat.meta_cat.MetaCAT, optional):
            A list of models that will be applied sequentially on each
            detected annotation.

    Attributes (limited):
        cdb (medcat.cdb.CDB):
            Concept database used with this CAT instance, please do not assign
            this value directly.
        vocab (medcat.utils.vocab.Vocab):
            The vocabulary object used with this instance, please do not assign
            this value directly.
        config - WILL BE REMOVED - TEMPORARY PLACEHOLDER

    Examples:
        >>>cat = CAT(cdb, vocab)
        >>>spacy_doc = cat("Put some text here")
        >>>print(spacy_doc.ents) # Detected entites
    '''
    def __init__(self, cdb, vocab=None, skip_stopwords=True, meta_cats=[], config={}, tokenizer=None):
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
        self.spacy_cat = SpacyCat(cdb=self.cdb, vocab=self.vocab, tokenizer=tokenizer)
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
        r'''
        Unlink a concept name from the CUI (or all CUIs if full_unlink), removes the link from
        the Concept Database (CDB). As a consequence medcat will never again link the `name`
        to this CUI - meaning the name will not be detected as a concept in the future.

        Args:
            cui (str):
                The CUI from which the `name` will be removed
            name (str):
                The span of text to be removed from the linking dictionary
            full_unlink (boolean):
                If True, the `name` will not only be removed from the given `cui` but from
                each concept in the database that is associated with this name.
        Examples:
            >>> # To never again link C0020538 to HTN
            >>> cat.unlink_concept_name('C0020538', 'htn', False)
        '''
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
        r'''
        Please do not use directly. This function will add a name to a CUI (existing or new).

        Args:
            cui (str):
                The CUI to which to add the name
            source_val (str):
                The `name` or span or source_value that will be linked to the cui
            is_pref_name (boolean):
                Is this source_val the prefered `name` for this CUI (concept)
            only_new (bool):
                Only add the name if it does not exist in the current CDB and is not linked
                to any concept (CUI) in the current CDB.
            desc (str):
                Description for this concept
            tui (str):
                Semenantic Type identifer for this concept, should be a TUI that exisit in the
                current CDB. Have a look at cdb.tui2names - for a list of all existing TUIs
                in the current CDB.

        Examples:
            Do not use.
        '''
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
        r'''
        This function will add a `name` (source_val) to a CUI (existing or new). It will teach medcat
        that this source_val is linked to this CUI.

        Args:
            cui (str):
                The CUI to which to add the name
            source_val (str):
                The `name` or span or source_value that will be linked to the cui
            text (str, optional):
                Text in which an example of this source_val can be found. Used for supervised/online
                training. This is basically one sample in a dataset for supervised training.
            is_pref_name (boolean):
                Is this source_val the prefered `name` for this CUI (concept)
            tkn_inds (list of ints, optional):
                Should be in the form: [3, 4, 5, ...]. This should be used only if you are providing a spacy_doc also.
                It gives the indicies of the tokens in a spacy document where the source_val can be found.
            text_inds (list, optional):
                A list that has only two values the start index for this `source_val` in the `text` and the end index.
                Used if you are not providing a spacy_doc. But are providing a `text` - it is optional and if not provided
                medcat will try to automatically find the start and end index.
            spacy_doc ()
            TODO:
            lr (float):
                The learning rate that will be used if you are providing the `text` that will be used for supervised/active
                learning.

            only_new (bool):
                Only add the name if it does not exist in the current CDB and is not linked
                to any concept (CUI) in the current CDB.
            desc (str):
                Description for this concept
            tui (str):
                Semenantic Type identifer for this concept, should be a TUI that exisit in the
                current CDB. Have a look at cdb.tui2names - for a list of all existing TUIs
                in the current CDB.

        Examples:
            Do not use.
        '''
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


    def _print_stats(self, data, epoch=0, use_filters=False, use_overlaps=False, use_cui_doc_limit=False,
                     use_groups=False):
        r'''
        Print metrics on a dataset (F1, P, R), it will also print the concepts that have the most FP,FN,TP.

        Args:
            data (list of dict):
                The json object that we get from MedCATtrainer on export.
            epoch (int):
                Used during training, so we know what epoch is it.
            use_filters (boolean):
                Each project in medcattrainer can have filters, do we want to respect those filters
                when calculating metrics.
            use_overlaps (boolean):
                Allow overlapping entites, nearly always False as it is very difficult to annotate overlapping entites.
            use_cui_doc_limit (boolean):
                If True the metrics for a CUI will be only calculated if that CUI appears in a document, in other words
                if the document was annotated for that CUI. Useful in very specific situations when during the annotation
                process the set of CUIs changed.
            use_groups (boolean):
                If True concepts that have groups will be combined and stats will be reported on groups.

        Returns:
            fps (dict):
                False positives for each CUI
            fns (dict):
                False negatives for each CUI
            tps (dict):
                True positives for each CUI
            cui_prec (dict):
                Precision for each CUI
            cui_rec (dict):
                Recall for each CUI
            cui_f1 (dict):
                F1 for each CUI
            cui_counts (dict):
                Number of occurrence for each CUI
        '''
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
        examples = {'fp': {}, 'fn': {}, 'tp': {}}

        fp_docs = set()
        fn_docs = set()
        if self.spacy_cat.TUI_FILTER is None:
            _tui_filter = None
        else:
            _tui_filter = list(self.spacy_cat.TUI_FILTER)
        if self.spacy_cat.CUI_FILTER is None:
            _cui_filter = None
        else:
            _cui_filter = list(self.spacy_cat.CUI_FILTER)

        for pind, project in tqdm(enumerate(data['projects']), desc="Stats project", total=len(data['projects']), leave=False):
            cui_filter = None
            tui_filter = None

            if use_filters:
                if 'cuis' in project and len(project['cuis'].strip()) > 0:
                    cui_filter = set([x.strip() for x in project['cuis'].split(",")])
                if 'tuis' in project and len(project['tuis'].strip()) > 0:
                    tui_filter = set([x.strip().upper() for x in project['tuis'].split(",")])

                self.spacy_cat.TUI_FILTER = tui_filter
                self.spacy_cat.CUI_FILTER = cui_filter

            start_time = time.time()
            for dind, doc in tqdm(enumerate(project['documents']), desc='Stats document', total=len(project['documents']), leave=False):
                spacy_doc = self(doc['text'])
                anns = doc['annotations']
                if use_overlaps:
                    p_anns = spacy_doc._.ents
                else:
                    p_anns = spacy_doc.ents

                anns_norm = []
                anns_norm_neg = []
                anns_examples = []
                anns_norm_cui = []
                for ann in anns:
                    if (cui_filter is None and tui_filter is None) or (cui_filter is not None and ann['cui'] in cui_filter) or \
                       (tui_filter is not None and self.cdb.cui2tui.get(ann['cui'], 'unk') in tui_filter):
                        cui = ann['cui']
                        if use_groups:
                            cui = self.cdb.cui2info.get(cui, {}).get("group", cui)

                        if ann.get('validated', True) and (not ann.get('killed', False) and not ann.get('deleted', False)):
                            anns_norm.append((ann['start'], cui))
                            anns_examples.append({"text": doc['text'][max(0, ann['start']-60):ann['end']+60],
                                                  "cui": cui,
                                                  "source value": ann['value'],
                                                  "acc": 1,
                                                  "project index": pind,
                                                  "document inedex": dind})
                        elif ann.get('validated', True) and (ann.get('killed', False) or ann.get('deleted', False)):
                            anns_norm_neg.append((ann['start'], cui))


                        if ann.get("validated", True):
                            # This is used to test was someone annotating for this CUI in this document
                            anns_norm_cui.append(cui)
                            cui_counts[cui] = cui_counts.get(cui, 0) + 1

                p_anns_norm = []
                p_anns_examples = []
                for ann in p_anns:
                    cui = ann._.cui
                    if use_groups:
                        cui = self.cdb.cui2info.get(cui, {}).get("group", cui)
                    p_anns_norm.append((ann.start_char, cui))
                    p_anns_examples.append({"text": doc['text'][max(0, ann.start_char-60):ann.end_char+60],
                                          "cui": cui,
                                          "source value": ann.text,
                                          "acc": float(ann._.acc),
                                          "project index": pind,
                                          "document inedex": dind})


                for iann, ann in enumerate(p_anns_norm):
                    if not use_cui_doc_limit or ann[1] in anns_norm_cui:
                        cui = ann[1]
                        if ann in anns_norm:
                            tp += 1
                            tps[cui] = tps.get(cui, 0) + 1

                            example = p_anns_examples[iann]
                            examples['tp'][cui] = examples['tp'].get(cui, []) + [example]
                        else:
                            fp += 1
                            fps[cui] = fps.get(cui, 0) + 1
                            fp_docs.add(doc['name'])

                            # Add example for this FP prediction
                            example = p_anns_examples[iann]
                            if ann in anns_norm_neg:
                                # Means that it really was annotated as negative
                                example['real_fp'] = True

                            examples['fp'][cui] = examples['fp'].get(cui, []) + [example]

                for iann, ann in enumerate(anns_norm):
                    if ann not in p_anns_norm:
                        cui = ann[1]
                        fn += 1
                        fn_docs.add(doc['name'])

                        fns[cui] = fns.get(cui, 0) + 1
                        examples['fn'][cui] = examples['fn'].get(cui, []) + [anns_examples[iann]]

        try:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = (prec + rec) / 2
            print("Epoch: {}, Prec: {}, Rec: {}, F1: {}\n".format(epoch, prec, rec, f1))
            print("Docs with false positives: {}\n".format("; ".join([str(x) for x in list(fp_docs)[0:10]])))
            print("Docs with false negatives: {}\n".format("; ".join([str(x) for x in list(fn_docs)[0:10]])))

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
                list(self.cdb.cui2original_names.get(cui, [cui]))[0]), cui, fps[cui]) for cui in list(fps.keys())[0:10]]
            pr_fns = [(self.cdb.cui2pretty_name.get(cui,
                list(self.cdb.cui2original_names.get(cui, [cui]))[0]), cui, fns[cui]) for cui in list(fns.keys())[0:10]]
            pr_tps = [(self.cdb.cui2pretty_name.get(cui,
                list(self.cdb.cui2original_names.get(cui, [cui]))[0]), cui, tps[cui]) for cui in list(tps.keys())[0:10]]


            print("\n\nFalse Positives\n")
            for one in pr_fps:
                print("{:70} - {:20} - {:10}".format(str(one[0])[0:69], str(one[1])[0:19], one[2]))
            print("\n\nFalse Negatives\n")
            for one in pr_fns:
                print("{:70} - {:20} - {:10}".format(str(one[0])[0:69], str(one[1])[0:19], one[2]))
            print("\n\nTrue Positives\n")
            for one in pr_tps:
                print("{:70} - {:20} - {:10}".format(str(one[0])[0:69], str(one[1])[0:19], one[2]))
            print("*"*110 + "\n")


        except Exception as e:
            traceback.print_exc()

        self.spacy_cat.TUI_FILTER = _tui_filter
        self.spacy_cat.CUI_FILTER = _cui_filter

        return fps, fns, tps, cui_prec, cui_rec, cui_f1, cui_counts, examples


    def train_supervised(self, data_path, reset_cdb=False, reset_cui_count=False, nepochs=30, lr=None,
                         anneal=None, print_stats=True, use_filters=False, terminate_last=False, use_overlaps=False,
                         use_cui_doc_limit=False, test_size=0, force_manually_created=False, use_groups=False,
                         never_terminate=False):
        r'''
        Run supervised training on a dataset from MedCATtrainer. Please take care that this is more a simiulated
        online training then supervised.

        Args:
            data_path (str):
                The path to the json file that we get from MedCATtrainer on export.
            reset_cdb (boolean):
                This will remove all concepts from the existing CDB and build a new CDB based on the
                concepts that appear in the training data. It will be impossible to get back the removed
                concepts.
            reset_cui_count (boolean):
                Used for training with weight_decay (annealing). Each concept has a count that is there
                from the beginning of the CDB, that count is used for annealing. Resetting the count will
                significantly incrase the training impact. This will reset the count only for concepts
                that exist in the the training data.
            nepochs (int):
                Number of epochs for which to run the training.
            lr (int):
                If set it will overwrite the global LR from config.
            anneal (boolean):
                If true annealing will be used when training.
            print_stats (boolean):
                If true stats will be printed during training (prints stats every 5 epochs).
            use_filters (boolean):
                Each project in medcattrainer can have filters, do we want to respect those filters
                when calculating metrics.
            terminate_last (boolean):
                If true, concept termination will be done after all training.
            use_overlaps (boolean):
                Allow overlapping entites, nearly always False as it is very difficult to annotate overlapping entites.
            use_cui_doc_limit (boolean):
                If True the metrics for a CUI will be only calculated if that CUI appears in a document, in other words
                if the document was annotated for that CUI. Useful in very specific situations when during the annotation
                process the set of CUIs changed.
            test_size (float):
                If > 0 the data set will be split into train test based on this ration. Should be between 0 and 1.
                Usually 0.1 is fine.
            force_manually_created (float):
                Check add_name for more details, if true all concepts in the dataset will be treated as manually
                created.
            use_groups (boolean):
                If True concepts that have groups will be combined and stats will be reported on groups.
            never_terminate (boolean):
                If True no termination will be applied

        Returns:
            fp (dict):
                False positives for each CUI
            fn (dict):
                False negatives for each CUI
            tp (dict):
                True positives for each CUI
            p (dict):
                Precision for each CUI
            r (dict):
                Recall for each CUI
            f1 (dict):
                F1 for each CUI
            cui_counts (dict):
                Number of occurrence for each CUI
            examples (dict):
                FP/FN examples of sentences for each CUI
        '''
        fp = fn = tp = p = r = f1 = cui_counts = examples = {}

        self.train = False
        data = json.load(open(data_path))
        cui_counts = {}

        if test_size == 0:
            test_set = data
            train_set = data
        else:
            train_set, test_set, _, _ = make_mc_train_test(data, self.cdb, test_size=test_size)

        if print_stats:
            self._print_stats(test_set, use_filters=use_filters, use_cui_doc_limit=use_cui_doc_limit, use_overlaps=use_overlaps,
                    use_groups=use_groups)

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
        if not never_terminate:
            for project in train_set['projects']:
                for doc in project['documents']:
                    for ann in doc['annotations']:
                        if ann.get('killed', False):
                            self.unlink_concept_name(ann['cui'], ann['value'])

        for epoch in tqdm(range(nepochs), desc='Epoch', leave=False):
            # Print acc before training
            for project in tqdm(train_set['projects'], desc='Project', leave=False, total=len(train_set['projects'])):
                for i_doc, doc in tqdm(enumerate(project['documents']), desc='Document', leave=False, total=len(project['documents'])):
                    spacy_doc = self(doc['text'])
                    for ann in doc['annotations']:
                        if not ann.get('killed', False):
                            cui = ann['cui']
                            start = ann['start']
                            end = ann['end']
                            deleted = ann.get('deleted', False)
                            manually_created = False
                            if force_manually_created or ann.get('manually_created', False) or ann.get('alternative', False):
                                manually_created = True
                            self.add_name(cui=cui,
                                          source_val=ann['value'],
                                          spacy_doc=spacy_doc,
                                          text_inds=[start, end],
                                          negative=deleted,
                                          lr=lr,
                                          anneal=anneal,
                                          manually_created=manually_created)
            if terminate_last and not never_terminate:
                # Remove entites that were terminated, but after all training is done
                for project in train_set['projects']:
                    for doc in project['documents']:
                        for ann in doc['annotations']:
                            if ann.get('killed', False):
                                self.unlink_concept_name(ann['cui'], ann['value'])

            if epoch % 5 == 0:
                if print_stats:
                    fp, fn, tp, p, r, f1, cui_counts, examples = self._print_stats(test_set, epoch=epoch+1,
                                                             use_filters=use_filters,
                                                             use_cui_doc_limit=use_cui_doc_limit,
                                                             use_overlaps=use_overlaps,
                                                             use_groups=use_groups)
        return fp, fn, tp, p, r, f1, cui_counts, examples

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
                out_ent['pretty_name'] = self.cdb.cui2pretty_name.get(cui, '')
                out_ent['cui'] = cui
                out_ent['tui'] = str(ent._.tui)
                out_ent['type'] = str(self.cdb.tui2name.get(out_ent['tui'], ''))
                out_ent['source_value'] = str(ent.text)
                out_ent['acc'] = str(ent._.acc)
                out_ent['start'] = ent.start_char
                out_ent['end'] = ent.end_char
                out_ent['info'] = self.cdb.cui2info.get(cui, {})
                out_ent['id'] = str(ent._.id)
                out_ent['meta_anns'] = {}

                if hasattr(ent._, 'meta_anns') and ent._.meta_anns:
                    for key in ent._.meta_anns.keys():
                        one = {'name': key, 'value': ent._.meta_anns[key]}
                        out_ent['meta_anns'][key] = one

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

    def add_cui_to_group(self, cui, group_name, reset_all_groups=False):
        r'''
        Ads a CUI to a group, will appear in cdb.cui2info['group']

        Args:
            cui (str):
                The concept to be added
            group_name (str):
                The group to whcih the concept will be added
            reset_all_groups (boolean):
                If True it will reset all existing groups and remove them.

        Examples:
            >>> cat.add_cui_to_group("S-17", 'pain')
        '''

        # Reset if needed
        if reset_all_groups:
            for _cui in self.cdb.cui2info.keys():
                _ = self.cdb.cui2info[_cui].pop('group', None)

        # Add
        if cui in self.cdb.cui2info:
            self.cdb.cui2info[cui]['group'] = group_name
        else:
            self.cdb.cui2info[cui] = {'group': group_name}
