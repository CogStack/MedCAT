import os
import shutil
import pickle
import traceback
import json
import logging
import math
import types
from copy import deepcopy
from multiprocessing import Process, Manager, Queue, cpu_count
from time import sleep
from typing import Union, List, Tuple, Optional, Dict, Iterable, Generator
from tqdm.autonotebook import tqdm
from spacy.tokens import Span, Doc

from medcat.utils.matutils import intersect_nonempty_set
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.pipe import Pipe
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.utils.loggers import add_handlers
from medcat.utils.data_utils import make_mc_train_test, get_false_positives
from medcat.utils.normalizers import BasicSpellChecker
from medcat.ner.vocab_based_ner import NER
from medcat.linking.context_based_linker import Linker
from medcat.utils.filters import get_project_filters, check_filters
from medcat.preprocessing.cleaners import prepare_name
from medcat.utils.helpers import tkns_from_doc
from medcat.meta_cat import MetaCAT
from medcat.utils.meta_cat.data_utils import json_to_fake_spacy


class CAT(object):
    r'''
    The main MedCAT class used to annotate documents, it is built on top of spaCy
    and works as a spaCy pipline. Creates an instance of a spaCy pipline that can
    be used as a spacy nlp model.

    Args:
        cdb (medcat.cdb.CDB):
            The concept database that will be used for NER+L
        config (medcat.config.Config):
            Global configuration for medcat
        vocab (medcat.vocab.Vocab, optional):
            Vocabulary used for vector embeddings and spelling. Default: None
        meta_cats (list of medcat.meta_cat.MetaCAT, optional):
            A list of models that will be applied sequentially on each
            detected annotation.

    Attributes (limited):
        cdb (medcat.cdb.CDB):
            Concept database used with this CAT instance, please do not assign
            this value directly.
        config (medcat.config.Config):
            The global configuration for medcat. Usually cdb.config can be used for this
            field.
        vocab (medcat.utils.vocab.Vocab):
            The vocabulary object used with this instance, please do not assign
            this value directly.
        config - WILL BE REMOVED - TEMPORARY PLACEHOLDER

    Examples:
        >>>cat = CAT(cdb, vocab)
        >>>spacy_doc = cat("Put some text here")
        >>>print(spacy_doc.ents) # Detected entites
    '''
    log = logging.getLogger(__package__)
    # Add file and console handlers
    log = add_handlers(log)

    def __init__(self, cdb, config, vocab, meta_cats=[]):
        self.cdb = cdb
        self.vocab = vocab
        # Take config from the cdb
        self.config = config

        # Set log level
        self.log.setLevel(self.config.general['log_level'])

        # Build the pipeline
        self.pipe = Pipe(tokenizer=spacy_split_all, config=self.config)
        self.pipe.add_tagger(tagger=tag_skip_and_punct,
                             name='skip_and_punct',
                             additional_fields=['is_punct'])

        spell_checker = BasicSpellChecker(cdb_vocab=self.cdb.vocab, config=self.config, data_vocab=vocab)
        self.pipe.add_token_normalizer(spell_checker=spell_checker, config=self.config)

        # Add NER
        self.ner = NER(self.cdb, self.config)
        self.pipe.add_ner(self.ner)

        # Add LINKER
        self.linker = Linker(self.cdb, vocab, self.config)
        self.pipe.add_linker(self.linker)

        # Add meta_annotaiton classes if they exist
        self._meta_annotations = False
        for meta_cat in meta_cats:
            self.pipe.add_meta_cat(meta_cat, meta_cat.config.general['category_name'])
            self._meta_annotations = True

        # Set max document length
        self.pipe.nlp.max_length = self.config.preprocessing.get('max_document_length')


    def get_spacy_nlp(self):
        ''' Returns the spacy pipeline with MedCAT
        '''
        return self.pipe.nlp


    def create_model_pack(self, save_dir_path, model_pack_name='medcat_model_pack'):
        r''' Will crete a .zip file containing all the models in the current running instance
        of MedCAT. This is not the most efficient way, for sure, but good enough for now.
        '''

        self.log.warning("This will save all models into a zip file, can take some time and require quite a bit of disk space.")
        _save_dir_path = save_dir_path
        save_dir_path = os.path.join(save_dir_path, model_pack_name)

        os.makedirs(save_dir_path, exist_ok=True)

        # Save the used spacy model
        spacy_path = os.path.join(save_dir_path, 'spacy_model')
        if str(self.pipe.nlp._path) != spacy_path:
            # First remove if something is there
            shutil.rmtree(spacy_path, ignore_errors=True)
            shutil.copytree(self.pipe.nlp._path, spacy_path)

        # Change the name of the spacy model in the config
        self.config.general['spacy_model'] = 'spacy_model'

        # Save the CDB
        cdb_path = os.path.join(save_dir_path, "cdb.dat")
        self.cdb.save(cdb_path)

        # Save the Vocab
        vocab_path = os.path.join(save_dir_path, "vocab.dat")
        self.vocab.save(vocab_path)

        # Save all meta_cats
        for comp in self.pipe.nlp.components:
            if isinstance(comp[1], MetaCAT):
                name = comp[0]
                meta_path = os.path.join(save_dir_path, "meta_" + name)
                comp[1].save(meta_path)

        shutil.make_archive(os.path.join(_save_dir_path, model_pack_name), 'zip', root_dir=save_dir_path)


    @classmethod
    def load_model_pack(cls, zip_path):
        r''' Load everything
        '''
        from medcat.cdb import CDB
        from medcat.vocab import Vocab
        from medcat.meta_cat import MetaCAT

        base_dir = os.path.dirname(zip_path)
        filename = os.path.basename(zip_path)
        foldername = filename.replace(".zip", '')

        model_pack_path = os.path.join(base_dir, foldername)
        if os.path.exists(model_pack_path):
            print("Found an existing unziped model pack at: {}, the provided zip will not be touched.".format(model_pack_path))
        else:
            print("Unziping the model pack and loading models.")
            shutil.unpack_archive(zip_path, extract_dir=model_pack_path)

        # Load the CDB
        cdb_path = os.path.join(model_pack_path, "cdb.dat")
        cdb = CDB.load(cdb_path)

        # Modify the config to contain full path to spacy model
        cdb.config.general['spacy_model'] = os.path.join(model_pack_path, cdb.config.general['spacy_model'])

        # Load Vocab
        vocab_path = os.path.join(model_pack_path, "vocab.dat")
        vocab = Vocab.load(vocab_path)

        # Find meta models in the model_pack
        meta_paths = [os.path.join(model_pack_path, path) for path in os.listdir(model_pack_path) if path.startswith('meta_')]
        meta_cats = []
        for meta_path in meta_paths:
            meta_cats.append(MetaCAT.load(meta_path))

        return cls(cdb=cdb, config=cdb.config, vocab=vocab, meta_cats=meta_cats)


    def __call__(self, text: str, do_train: bool = False):
        r'''
        Push the text through the pipeline.

        Args:
            text (string/Iterable):
                The text or the sequence of texts or the sequence of (id, text) to be annotated, if the text length is longer
                than self.config.preprocessing['max_document_length'] it will be trimmed to that length.
            do_train (bool, defaults to `False`):
                This causes so many screwups when not there, so I'll force training
                to False. To run training it is much better to use the self.train() function
                but for some special cases I'm leaving it here also.
        Returns:
            A single spacy document or multiple spacy documents with the extracted entities
        '''
        # Should we train - do not use this for training, unless you know what you are doing. Use the
        #self.train() function
        self.config.linking['train'] = do_train

        if text is None:
            self.log.error("The input text should be either a string or a sequence of strings but got %s", type(text))
            return None
        else:
            text = self._get_trimmed_text(str(text))
            return self.pipe(text)

    def _print_stats(self, data, epoch=0, use_project_filters=False, use_overlaps=False, use_cui_doc_limit=False,
                     use_groups=False, extra_cui_filter=None):
        r''' TODO: Refactor and make nice
        Print metrics on a dataset (F1, P, R), it will also print the concepts that have the most FP,FN,TP.

        Args:
            data (list of dict):
                The json object that we get from MedCATtrainer on export.
            epoch (int):
                Used during training, so we know what epoch is it.
            use_project_filters (boolean):
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
            extra_cui_filter(boolean):
                This filter will be intersected with all other filters, or if all others are not set then only this one will be used.

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
            examples (dict):
                Examples for each of the fp, fn, tp. Format will be examples['fp']['cui'][<list_of_examples>]
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
        # Reset and shortcut for filters
        filters = self.config.linking['filters']
        for pind, project in tqdm(enumerate(data['projects']), desc="Stats project", total=len(data['projects']), leave=False):
            filters['cuis'] = set()

            # Add extrafilter if set
            if isinstance(extra_cui_filter, set):
                filters['cuis'] = extra_cui_filter

            if use_project_filters:
                project_filter = get_project_filters(cuis=project.get('cuis', None),
                                                      type_ids=project.get('tuis', None),
                                                      cdb=self.cdb)
                # Intersect project filter with existing if it has something
                if project_filter:
                    filters['cuis'] = intersect_nonempty_set(project_filter, filters['cuis'])

            for dind, doc in tqdm(
                enumerate(project["documents"]),
                desc="Stats document",
                total=len(project["documents"]),
                leave=False,
            ):
                anns = self._get_doc_annotations(doc)

                # Apply document level filtering, in this case project_filter is ignored while the extra_cui_filter is respected still
                if use_cui_doc_limit:
                    _cuis = set([ann['cui'] for ann in anns])
                    if _cuis:
                        filters['cuis'] = intersect_nonempty_set(_cuis, extra_cui_filter)
                    else:
                        filters['cuis'] = {'empty'}

                spacy_doc = self(doc['text'])

                if use_overlaps:
                    p_anns = spacy_doc._.ents
                else:
                    p_anns = spacy_doc.ents

                anns_norm = []
                anns_norm_neg = []
                anns_examples = []
                anns_norm_cui = []
                for ann in anns:
                    cui = ann['cui']
                    if check_filters(cui, filters):
                        if use_groups:
                            cui = self.cdb.addl_info['cui2group'].get(cui, cui)

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
                        cui = self.cdb.addl_info['cui2group'].get(cui, cui)

                    p_anns_norm.append((ann.start_char, cui))
                    p_anns_examples.append({"text": doc['text'][max(0, ann.start_char-60):ann.end_char+60],
                                          "cui": cui,
                                          "source value": ann.text,
                                          "acc": float(ann._.context_similarity),
                                          "project index": pind,
                                          "document inedex": dind})


                for iann, ann in enumerate(p_anns_norm):
                    cui = ann[1]
                    if ann in anns_norm:
                        tp += 1
                        tps[cui] = tps.get(cui, 0) + 1

                        example = p_anns_examples[iann]
                        examples['tp'][cui] = examples['tp'].get(cui, []) + [example]
                    else:
                        fp += 1
                        fps[cui] = fps.get(cui, 0) + 1
                        fp_docs.add(doc.get('name', 'unk'))

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
                        fn_docs.add(doc.get('name', 'unk'))

                        fns[cui] = fns.get(cui, 0) + 1
                        examples['fn'][cui] = examples['fn'].get(cui, []) + [anns_examples[iann]]

        try:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2*(prec*rec) / (prec + rec)
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
                f1 = 2*(prec*rec) / (prec + rec)
                cui_prec[cui] = prec
                cui_rec[cui] = rec
                cui_f1[cui] = f1


            # Get top 10
            pr_fps = [(self.cdb.cui2preferred_name.get(cui,
                list(self.cdb.cui2names.get(cui, [cui]))[0]), cui, fps[cui]) for cui in list(fps.keys())[0:10]]
            pr_fns = [(self.cdb.cui2preferred_name.get(cui,
                list(self.cdb.cui2names.get(cui, [cui]))[0]), cui, fns[cui]) for cui in list(fns.keys())[0:10]]
            pr_tps = [(self.cdb.cui2preferred_name.get(cui,
                list(self.cdb.cui2names.get(cui, [cui]))[0]), cui, tps[cui]) for cui in list(tps.keys())[0:10]]


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

        except Exception:
            traceback.print_exc()

        return fps, fns, tps, cui_prec, cui_rec, cui_f1, cui_counts, examples

    def train(self, data_iterator, fine_tune=True, progress_print=1000):
        """ Runs training on the data, note that the maximum length of a line
        or document is 1M characters. Anything longer will be trimmed.

        data_iterator:
            Simple iterator over sentences/documents, e.g. a open file
            or an array or anything that we can use in a for loop.
        fine_tune:
            If False old training will be removed
        progress_print:
            Print progress after N lines
        """
        if not fine_tune:
            self.log.info("Removing old training data!")
            self.cdb.reset_training()

        cnt = 0
        for line in data_iterator:
            if line is not None and line:
                # Convert to string
                line = str(line).strip()

                try:
                    _ = self(line, do_train=True)
                except Exception as e:
                    self.log.warning("LINE: '%s...' \t WAS SKIPPED", line[0:100])
                    self.log.warning("BECAUSE OF: %s", str(e))
                if cnt % progress_print == 0:
                    self.log.info("DONE: %s", str(cnt))
                cnt += 1

        self.config.linking['train'] = False

    def add_cui_to_group(self, cui, group_name):
        r'''
        Ads a CUI to a group, will appear in cdb.addl_info['cui2group']

        Args:
            cui (str):
                The concept to be added
            group_name (str):
                The group to whcih the concept will be added

        Examples:
            >>> cat.add_cui_to_group("S-17", 'pain')
        '''

        # Add group_name
        self.cdb.addl_info['cui2group'][cui] = group_name

    def unlink_concept_name(self, cui, name, preprocessed_name=False):
        r'''
        Unlink a concept name from the CUI (or all CUIs if full_unlink), removes the link from
        the Concept Database (CDB). As a consequence medcat will never again link the `name`
        to this CUI - meaning the name will not be detected as a concept in the future.

        Args:
            cui (str):
                The CUI from which the `name` will be removed
            name (str):
                The span of text to be removed from the linking dictionary
        Examples:
            >>> # To never again link C0020538 to HTN
            >>> cat.unlink_concept_name('C0020538', 'htn', False)
        '''

        cuis = [cui]
        if preprocessed_name:
            names = {name: 'nothing'}
        else:
            names = prepare_name(name, self, {}, self.config)

        # If full unlink find all CUIs
        if self.config.general.get('full_unlink', False):
            for n in names:
                cuis.extend(self.cdb.name2cuis.get(n, []))

        # Remove name from all CUIs
        for c in cuis:
            self.cdb.remove_names(cui=c, names=names)

    def add_and_train_concept(self, cui, name, spacy_doc=None, spacy_entity=None, ontologies=set(), name_status='A', type_ids=set(),
                              description='', full_build=True, negative=False, devalue_others=False, do_add_concept=True):
        r''' Add a name to an existing concept, or add a new concept, or do not do anything if the name or concept already exists. Perform
        training if spacy_entity and spacy_doc are set.

        Args:
            cui (str):
                CUI of the concept
            name (str):
                Name to be linked to the concept (in the case of MedCATtrainer this is simply the
                selected value in text, no preprocessing or anything needed).
            spacy_doc (spacy.tokens.Doc):
                Spacy represenation of the document that was manually annotated.
            spacy_entity (List[spacy.tokens.Token]):
                Given the spacy document, this is the annotated span of text - list of annotated tokens that are marked with this CUI.
            negative (bool):
                Is this a negative or positive example.
            devalue_others:
                If set, cuis to which this name is assigned and are not `cui` will receive negative training given
                that negative=False.

            **other:
                Refer to CDB.add_concept
        '''

        names = prepare_name(name, self, {}, self.config)
        if do_add_concept:
            self.cdb.add_concept(cui=cui, names=names, ontologies=ontologies, name_status=name_status, type_ids=type_ids, description=description,
                                 full_build=full_build)

        if spacy_entity is not None and spacy_doc is not None:
            # Train Linking
            self.linker.context_model.train(cui=cui, entity=spacy_entity, doc=spacy_doc, negative=negative, names=names)

            if not negative and devalue_others:
                # Find all cuis
                cuis = set()
                for n in names:
                    cuis.update(self.cdb.name2cuis.get(n, []))
                # Remove the cui for which we just added positive training
                if cui in cuis:
                    cuis.remove(cui)
                # Add negative training for all other CUIs that link to these names
                for _cui in cuis:
                    self.linker.context_model.train(cui=_cui, entity=spacy_entity, doc=spacy_doc, negative=True)

    def train_supervised(self, data_path, reset_cui_count=False, nepochs=1,
                         print_stats=0, use_filters=False, terminate_last=False, use_overlaps=False,
                         use_cui_doc_limit=False, test_size=0, devalue_others=False, use_groups=False,
                         never_terminate=False, train_from_false_positives=False, extra_cui_filter=None):
        r''' TODO: Refactor, left from old
        Run supervised training on a dataset from MedCATtrainer. Please take care that this is more a simulated
        online training then supervised.

        Args:
            data_path (str):
                The path to the json file that we get from MedCATtrainer on export.
            reset_cui_count (boolean):
                Used for training with weight_decay (annealing). Each concept has a count that is there
                from the beginning of the CDB, that count is used for annealing. Resetting the count will
                significantly increase the training impact. This will reset the count only for concepts
                that exist in the the training data.
            nepochs (int):
                Number of epochs for which to run the training.
            print_stats (int):
                If > 0 it will print stats every print_stats epochs.
            use_filters (boolean):
                Each project in medcattrainer can have filters, do we want to respect those filters
                when calculating metrics.
            terminate_last (boolean):
                If true, concept termination will be done after all training.
            use_overlaps (boolean):
                Allow overlapping entities, nearly always False as it is very difficult to annotate overlapping entities.
            use_cui_doc_limit (boolean):
                If True the metrics for a CUI will be only calculated if that CUI appears in a document, in other words
                if the document was annotated for that CUI. Useful in very specific situations when during the annotation
                process the set of CUIs changed.
            test_size (float):
                If > 0 the data set will be split into train test based on this ration. Should be between 0 and 1.
                Usually 0.1 is fine.
            devalue_others(bool):
                Check add_name for more details.
            use_groups (boolean):
                If True concepts that have groups will be combined and stats will be reported on groups.
            never_terminate (boolean):
                If True no termination will be applied
            train_from_false_positives (boolean):
                If True it will use false positive examples detected by medcat and train from them as negative examples.
            extra_cui_filter(boolean):
                This filter will be intersected with all other filters, or if all others are not set then only this one will be used.

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
        # Backup filters
        _filters = deepcopy(self.config.linking['filters'])
        filters = self.config.linking['filters']

        fp = fn = tp = p = r = f1 = cui_counts = examples = {}
        with open(data_path) as f:
            data = json.load(f)
        cui_counts = {}

        if test_size == 0:
            self.log.info("Running without a test set, or train==test")
            test_set = data
            train_set = data
        else:
            train_set, test_set, _, _ = make_mc_train_test(data, self.cdb, test_size=test_size)

        if print_stats > 0:
            fp, fn, tp, p, r, f1, cui_counts, examples = self._print_stats(test_set, use_project_filters=use_filters,
                    use_cui_doc_limit=use_cui_doc_limit, use_overlaps=use_overlaps,
                    use_groups=use_groups, extra_cui_filter=extra_cui_filter)

        if reset_cui_count:
            # Get all CUIs
            cuis = []
            for project in train_set['projects']:
                for doc in project['documents']:
                    doc_annotations = self._get_doc_annotations(doc)
                    for ann in doc_annotations:
                        cuis.append(ann['cui'])
            for cui in set(cuis):
                if cui in self.cdb.cui2count_train:
                    self.cdb.cui2count_train[cui] = 10

        # Remove entities that were terminated
        if not never_terminate:
            for project in train_set['projects']:
                for doc in project['documents']:
                    doc_annotations = self._get_doc_annotations(doc)
                    for ann in doc_annotations:
                        if ann.get('killed', False):
                            self.unlink_concept_name(ann['cui'], ann['value'])
        for epoch in tqdm(range(nepochs), desc='Epoch', leave=False):
            # Print acc before training
            for project in tqdm(train_set['projects'], desc='Project', leave=False, total=len(train_set['projects'])):
                # Set filters in case we are using the train_from_fp
                filters['cuis'] = set()
                if isinstance(extra_cui_filter, set):
                    filters['cuis'] = extra_cui_filter

                if use_filters:
                    project_filter = get_project_filters(cuis=project.get('cuis', None),
                            type_ids=project.get('tuis', None),
                            cdb=self.cdb)

                    if project_filter:
                        filters['cuis'] = intersect_nonempty_set(project_filter, filters['cuis'])

                for _, doc in tqdm(enumerate(project['documents']), desc='Document', leave=False, total=len(project['documents'])):
                    spacy_doc = self(doc['text'])
                    # Compatibility with old output where annotations are a list
                    doc_annotations = self._get_doc_annotations(doc)
                    for ann in doc_annotations:
                        if not ann.get('killed', False):
                            cui = ann['cui']
                            start = ann['start']
                            end = ann['end']
                            spacy_entity = tkns_from_doc(spacy_doc=spacy_doc, start=start, end=end)
                            deleted = ann.get('deleted', False)
                            self.add_and_train_concept(cui=cui,
                                          name=ann['value'],
                                          spacy_doc=spacy_doc,
                                          spacy_entity=spacy_entity,
                                          negative=deleted,
                                          devalue_others=devalue_others)
                    if train_from_false_positives:
                        fps = get_false_positives(doc, spacy_doc)

                        for fp in fps:
                            self.add_and_train_concept(cui=fp._.cui,
                                                       name=fp.text,
                                                       spacy_doc=spacy_doc,
                                                       spacy_entity=fp,
                                                       negative=True,
                                                       do_add_concept=False)

            if terminate_last and not never_terminate:
                # Remove entities that were terminated, but after all training is done
                for project in train_set['projects']:
                    for doc in project['documents']:
                        doc_annotations = self._get_doc_annotations(doc)
                        for ann in doc_annotations:
                            if ann.get('killed', False):
                                self.unlink_concept_name(ann['cui'], ann['value'])

            if print_stats > 0 and (epoch + 1) % print_stats == 0:
                fp, fn, tp, p, r, f1, cui_counts, examples = self._print_stats(test_set, epoch=epoch+1,
                                                         use_project_filters=use_filters,
                                                         use_cui_doc_limit=use_cui_doc_limit,
                                                         use_overlaps=use_overlaps,
                                                         use_groups=use_groups,
                                                         extra_cui_filter=extra_cui_filter)
        # Set the filters again
        self.config.linking['filters'] = _filters
        return fp, fn, tp, p, r, f1, cui_counts, examples

    def get_entities(self,
                     text: str,
                     only_cui: bool = False,
                     addl_info: List[str] = ['cui2icd10', 'cui2ontologies', 'cui2snomed']) -> Dict:
        doc = self(text)
        out = self._doc_to_out(doc, only_cui, addl_info)
        return out

    def get_entities_multi_texts(self,
                     texts: Union[Iterable[str], Iterable[Tuple]],
                     only_cui: bool = False,
                     addl_info: List[str] = ['cui2icd10', 'cui2ontologies', 'cui2snomed'],
                     n_process: Optional[int] = None,
                     batch_size: Optional[int] = None) -> List[Union[Dict, None]]:
        r''' Get entities
        text:  text to be annotated
        return:  entities
        '''
        out: List[Union[Dict, None]]

        if n_process is None:
            out = []
            docs = self(self._generate_trimmed_texts(texts))
            for doc in docs:
                out.append(self._doc_to_out(doc, only_cui, addl_info))
        else:
            out = []
            self.pipe.set_error_handler(self._pipe_error_handler)
            try:
                texts = self._get_trimmed_texts(texts)
                docs = self.pipe.batch_multi_process(texts, n_process, batch_size)

                for doc in tqdm(docs, total=len(texts)):
                    doc = None if doc.text.strip() == '' else doc
                    out.append(self._doc_to_out(doc, only_cui, addl_info, out_with_text=True))

                # Currently spaCy cannot mark which pieces of texts failed within the pipe so be this workaround,
                # which also assumes texts are different from each others.
                if len(out) < len(texts):
                    self.log.warning("Found at least one failed batch and set output for enclosed texts to empty")
                    for i, text in enumerate(texts):
                        if i == len(out):
                            out.append(self._doc_to_out(None, only_cui, addl_info))
                        elif out[i]['text'] != text:
                            out.insert(i, self._doc_to_out(None, only_cui, addl_info))

                cnf_annotation_output = getattr(self.config, 'annotation_output', {})
                if not(cnf_annotation_output.get('include_text_in_output', False)):
                    for o in out:
                        o.pop('text', None)
            finally:
                self.pipe.reset_error_handler()

        return out

    def get_json(self, text, only_cui=False, addl_info=['cui2icd10', 'cui2ontologies']):
        """ Get output in json format

        text:  text to be annotated
        return:  json with fields {'entities': <>, 'text': text}
        """
        ents = self.get_entities(text, only_cui, addl_info=addl_info)['entities']
        out = {'annotations': ents, 'text': text}

        return json.dumps(out)

    def _separate_nn_components(self):
        # Loop though the models and check are there GPU devices
        nn_components = []
        for component in self.pipe.nlp.components:
            if isinstance(component[1], MetaCAT):
                self.pipe.nlp.disable_pipe(component[0])
                nn_components.append(component)

        return nn_components

    def _run_nn_components(self, docs, nn_components, id2text):
        r''' This will add meta_anns in-place to the docs dict.
        '''
        self.log.debug("Running GPU components separately")

        # First convert the docs into the fake spacy doc format
        spacy_docs = json_to_fake_spacy(docs, id2text=id2text)
        # Disable component locks also
        for name, component in nn_components:
            component.config.general['disable_component_lock'] = True

        for name, component in nn_components:
            spacy_docs = component.pipe(spacy_docs)

        for spacy_doc in spacy_docs:
            for ent in spacy_doc.ents:
                docs[spacy_doc.id]['entities'][ent._.id]['meta_anns'].update(ent._.meta_anns)

    def _batch_generator(self, data, batch_size_chars, skip_ids=set()):
        docs = []
        char_count = 0
        for doc in data:
            if doc[0] not in skip_ids:
                char_count += len(str(doc[1]))
                docs.append(doc)
                if char_count < batch_size_chars:
                    continue
                yield docs
                docs = []
                char_count = 0

        if len(docs) > 0:
            yield docs

    def _save_docs_to_file(self, docs, annotated_ids, save_dir_path, annotated_ids_path, part_counter=0):
        path = os.path.join(save_dir_path, 'part_{}.pickle'.format(part_counter))
        pickle.dump(docs, open(path, "wb"))
        self.log.info("Saved part: %s, to: %s", (part_counter, path))
        part_counter = part_counter + 1 # Increase for save, as it should be what is the next part
        pickle.dump((annotated_ids, part_counter), open(annotated_ids_path, 'wb'))
        return part_counter

    def multiprocessing(self,
                        data: Union[List[Tuple], Iterable[Tuple]],
                        nproc: int = 2,
                        batch_size_chars: int = 5000 * 1000,
                        only_cui: bool = False,
                        addl_info: List[str] = [],
                        separate_nn_components: bool = True,
                        out_split_size: int = None,
                        save_dir_path: str = None,) -> Dict:
        r''' Run multiprocessing for inference, if out_save_path and out_split_size is used this will also continue annotating
        documents if something is saved in that directory.

        Args:
            data(``):
                Iterator or array with format: [(id, text), (id, text), ...]
            nproc (`int`, defaults to 8):
                Number of processors
            batch_size_chars (`int`, defaults to 1000000):
                Size of a batch in number of characters, this should be around: NPROC * average_document_length * 200
            separate_nn_components (`bool`, defaults to True):
                If set the medcat pipe will be broken up into NN and not-NN components and
                they will be run sequentially. This is useful as the NN components
                have batching and like to process many docs at once, while the rest of the pipeline
                runs the documents one by one.
            out_split_size (`int`, None):
                If set once more than out_split_size documents are annotated
                they will be saved to a file (save_dir_path) and the memory cleared.
            save_dir_path(`str`, None):
                Where to save the annotated documents if splitting.

        Returns:
            A dictionary: {id: doc_json, id2: doc_json2, ...}, in case out_split_size is used
            the last batch will be returned while that and all previous batches will be
            written to disk (out_save_dir).
        '''
        if self._meta_annotations and not separate_nn_components:
            # Hack for torch using multithreading, which is not good if not 
            #separate_nn_components, need for CPU runs only
            import torch
            torch.set_num_threads(1)

        nn_components = []
        if separate_nn_components:
            nn_components = self._separate_nn_components()

        if save_dir_path is not None:
            os.makedirs(save_dir_path, exist_ok=True)

        internal_batch_size_chars = batch_size_chars // (5 * nproc)

        annotated_ids_path = os.path.join(save_dir_path, 'annotated_ids.pickle') if save_dir_path is not None else None
        if annotated_ids_path is not None and os.path.exists(annotated_ids_path):
            annotated_ids, part_counter = pickle.load(open(annotated_ids_path, 'rb'))
        else:
            annotated_ids = []
            part_counter = 0

        docs = {}
        for batch in self._batch_generator(data, batch_size_chars, skip_ids=set(annotated_ids)):
            self.log.info("Annotated until now: %s docs; Current BS: %s docs", (len(annotated_ids), len(batch)))
            try:
                _docs = self._multiprocessing_batch(data=batch,
                                                    nproc=nproc,
                                                    batch_size_chars=internal_batch_size_chars,
                                                    only_cui=only_cui,
                                                    addl_info=addl_info,
                                                    nn_components=nn_components)
                docs.update(_docs)
                annotated_ids.extend(_docs.keys())
                del _docs
                if out_split_size is not None and len(docs) > out_split_size:
                    # Save to file and reset the docs 
                    part_counter = self._save_docs_to_file(docs=docs,
                                           annotated_ids=annotated_ids,
                                           save_dir_path=save_dir_path,
                                           annotated_ids_path=annotated_ids_path,
                                           part_counter=part_counter)
                    del docs
                    docs = {}
            except Exception as e:
                self.log.warning("Failed an outer batch in the multiprocessing script")
                self.log.warning(e, exc_info=True, stack_info=True)

        # Save the last batch
        if out_split_size is not None and len(docs) > 0:
            # Save to file and reset the docs 
            self._save_docs_to_file(docs=docs,
                                   annotated_ids=annotated_ids,
                                   save_dir_path=save_dir_path,
                                   annotated_ids_path=annotated_ids_path,
                                   part_counter=part_counter)

        # Enable the GPU Components again
        if separate_nn_components:
            for name, _ in nn_components:
                # No need to do anything else as it was already in the pipe
                self.pipe.nlp.enable_pipe(name)

        return docs

    def _multiprocessing_batch(self,
                               data: Union[List[Tuple], Iterable[Tuple]],
                               nproc: int = 8,
                               batch_size_chars: int = 1000000,
                               only_cui: bool = False,
                               addl_info: List[str] = [],
                               nn_components=[]) -> Dict:
        r''' Run multiprocessing on one batch

        Args:
            data(``):
                Iterator or array with format: [(id, text), (id, text), ...]
            nproc (`int`, defaults to 8):
                Number of processors
            batch_size_chars (`int`, defaults to 1000000):
                Size of a batch in number of characters

        Returns:
            A dictionary: {id: doc_json, id2: doc_json2, ...}
        '''
        # Create the input output for MP
        in_q = Queue(maxsize=4*nproc)
        manager = Manager()
        out_dict = manager.dict()
        out_dict['processed'] = []

        # Create processes
        procs = []
        for i in range(nproc):
            p = Process(target=self._mp_cons,
                        kwargs={'in_q': in_q,
                                'out_dict': out_dict,
                                'pid': i,
                                'only_cui': only_cui,
                                'addl_info': addl_info})
            p.start()
            procs.append(p)

        id2text = {}
        for batch in self._batch_generator(data, batch_size_chars):
            if nn_components:
                # We need this for the json_to_fake_spacy
                id2text.update({k:v for k,v in batch})
            in_q.put(batch)

        # Final data point for workers
        for _ in range(nproc):
            in_q.put(None)
        # Join processes
        for p in procs:
            p.join()

        docs = {}
        for key in out_dict.keys():
            if 'pid' in key:
                # Covnerts a touple into a dict
                docs.update({k:v for k,v in out_dict[key]})

        # Cleanup - to prevent memory leaks, maybe
        out_dict.clear()
        del out_dict
        in_q.close()

        # If we have separate GPU components now we pipe that
        if nn_components:
            try:
                self._run_nn_components(docs, nn_components, id2text=id2text)
            except Exception as e:
                self.log.warning(e, exc_info=True, stack_info=True)

        return docs

    def multiprocessing_pipe(self,
                             in_data: Union[List[Tuple], Iterable[Tuple]],
                             nproc: Optional[int] = None,
                             batch_size: Optional[int] = None,
                             only_cui: bool = False,
                             addl_info: List[str] = [],
                             return_dict: bool = True,
                             batch_factor: int = 2) -> Union[List[Tuple], Dict]:
        r''' Run multiprocessing NOT FOR TRAINING

        in_data:  a list with format: [(id, text), (id, text), ...]
        nproc:  the number of processors
        batch_size: the number of texts to buffer
        return_dict: a flag for returning either a dict or a list of tuples

        return:  a dict: {id: doc_json, id: doc_json, ...} or if return_dict is False, a list of tuples: [(id, doc_json), (id, doc_json), ...]
        '''
        out: Union[Dict, List[Tuple]]

        if nproc == 0:
            raise ValueError("nproc cannot be set to zero")

        if self._meta_annotations:
            # Hack for torch using multithreading, which is not good here
            import torch
            torch.set_num_threads(1)

        in_data = list(in_data) if isinstance(in_data, types.GeneratorType) else in_data
        n_process = nproc if nproc is not None else min(max(cpu_count() - 1, 1), math.ceil(len(in_data) / batch_factor))
        batch_size = batch_size if batch_size is not None else math.ceil(len(in_data) / (batch_factor * abs(n_process)))

        entities = self.get_entities_multi_texts(texts=in_data, only_cui=only_cui, addl_info=addl_info,
                                     n_process=n_process, batch_size=batch_size)

        if return_dict:
            out = {}
            for idx, data in enumerate(in_data):
                out[data[0]] = entities[idx]
        else:
            out = []
            for idx, data in enumerate(in_data):
                out.append((data[0], entities[idx]))

        return out

    def _mp_cons(self, in_q, out_dict, pid=0, only_cui=False, addl_info=[]):
        out = []
        while True:
            if not in_q.empty():
                data = in_q.get()
                if data is None:
                    out_dict['pid: {}'.format(pid)] = out
                    break

                for i_text, text in data:
                    try:
                        # Annotate document
                        doc = self.get_entities(text=text, only_cui=only_cui, addl_info=addl_info)
                        out.append((i_text, doc))
                    except Exception as e:
                        self.log.warning("Exception in _mp_cons")
                        self.log.warning(e, exc_info=True, stack_info=True)

            sleep(1)

    def _doc_to_out(self,
                    doc: Doc,
                    only_cui: bool,
                    addl_info: List[str],
                    out_with_text: bool = False) -> Dict:
        out: Dict = {'entities': {}, 'tokens': []}
        cnf_annotation_output = getattr(self.config, 'annotation_output', {})
        if doc is not None:
            out_ent = {}
            if self.config.general.get('show_nested_entities', False):
                _ents = []
                for _ent in doc._.ents:
                    entity = Span(doc, _ent['start'], _ent['end'], label=_ent['label'])
                    entity._.cui = _ent['cui']
                    entity._.detected_name = _ent['detected_name']
                    entity._.context_similarity = _ent['context_similarity']
                    entity._.id = _ent['id']
                    if 'meta_anns' in _ent:
                        entity._.meta_anns = _ent['meta_anns']
                    _ents.append(entity)
            else:
                _ents = doc.ents

            if cnf_annotation_output.get("lowercase_context", True):
                doc_tokens = [tkn.text_with_ws.lower() for tkn in list(doc)]
            else:
                doc_tokens = [tkn.text_with_ws for tkn in list(doc)]

            if cnf_annotation_output.get('doc_extended_info', False):
                # Add tokens if extended info
                out['tokens'] = doc_tokens

            context_left = cnf_annotation_output.get('context_left', -1)
            context_right = cnf_annotation_output.get('context_right', -1)
            doc_extended_info = cnf_annotation_output.get('doc_extended_info', False)

            for _, ent in enumerate(_ents):
                cui = str(ent._.cui)
                if not only_cui:
                    out_ent['pretty_name'] = self.cdb.get_name(cui)
                    out_ent['cui'] = cui
                    out_ent['type_ids'] = list(self.cdb.cui2type_ids.get(cui, ''))
                    out_ent['types'] = [self.cdb.addl_info['type_id2name'].get(tui, '') for tui in out_ent['type_ids']]
                    out_ent['source_value'] = ent.text
                    out_ent['detected_name'] = str(ent._.detected_name)
                    out_ent['acc'] = float(ent._.context_similarity)
                    out_ent['context_similarity'] = float(ent._.context_similarity)
                    out_ent['start'] = ent.start_char
                    out_ent['end'] = ent.end_char
                    for addl in addl_info:
                        tmp = self.cdb.addl_info.get(addl, {}).get(cui, [])
                        out_ent[addl.split("2")[-1]] = list(tmp) if type(tmp) == set else tmp
                    out_ent['id'] = ent._.id
                    out_ent['meta_anns'] = {}

                    if doc_extended_info:
                        out_ent['start_tkn'] = ent.start
                        out_ent['end_tkn'] = ent.end

                    if context_left > 0 and context_right > 0:
                        out_ent['context_left'] = doc_tokens[max(ent.start - context_left, 0):ent.start]
                        out_ent['context_right'] = doc_tokens[ent.end:min(ent.end + context_right, len(doc_tokens))]
                        out_ent['context_center'] = doc_tokens[ent.start:ent.end]

                    if hasattr(ent._, 'meta_anns') and ent._.meta_anns:
                        out_ent['meta_anns'] = ent._.meta_anns

                    out['entities'][out_ent['id']] = dict(out_ent)
                else:
                    out['entities'][ent._.id] = cui

            if cnf_annotation_output.get('include_text_in_output', False) or out_with_text:
                out['text'] = doc.text
        return out

    def _get_trimmed_text(self, text: str) -> Optional[str]:
        return text[0:self.config.preprocessing.get('max_document_length')] if text is not None and len(text) > 0 else ""

    def _generate_trimmed_texts(self, texts: Union[Iterable[str], Iterable[Tuple]]) -> Generator[Optional[str], None, None]:
        for text in texts:
            if isinstance(text, Tuple):
                yield self._get_trimmed_text(text[1])
            else:
                yield self._get_trimmed_text(text)

    def _get_trimmed_texts(self, texts: Union[Iterable[str], Iterable[Tuple]]) -> Iterable[Union[str, None]]:
        trimmed = []
        for text in texts:
            if isinstance(text, Tuple):
                trimmed.append(self._get_trimmed_text(text[1]))
            else:
                trimmed.append(self._get_trimmed_text(text))
        return trimmed

    @staticmethod
    def _pipe_error_handler(proc_name, proc, docs, e):
        CAT.log.warning("Exception raised when applying component %s to a batch of docs.", proc_name)
        CAT.log.warning(e, exc_info=True, stack_info=True)
        if docs is not None:
            CAT.log.warning("Docs contained in the batch:")
            for doc in docs:
                if hasattr(doc, "text"):
                    CAT.log.warning("%s...", doc.text[:50])

    @staticmethod
    def _get_doc_annotations(doc):
        if type(doc['annotations']) == list:
            return doc['annotations']
        if type(doc['annotations']) == dict:
            return doc['annotations'].values()
        return None

    def destroy_pipe(self):
        self.pipe.destroy()
