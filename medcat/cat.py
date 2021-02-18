import traceback
import json
import time
import logging
from functools import partial
from copy import deepcopy
from tqdm.autonotebook import tqdm
from multiprocessing import Process, Manager, Queue, Pool, Array
from time import sleep

from medcat.cdb import CDB
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.pipe import Pipe
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.utils.loggers import add_handlers
from medcat.utils.data_utils import make_mc_train_test
from medcat.utils.normalizers import BasicSpellChecker
from medcat.ner.vocab_based_ner import NER
from medcat.linking.context_based_linker import Linker
from medcat.utils.filters import process_old_project_filters, check_filters
from medcat.preprocessing.cleaners import prepare_name
from medcat.utils.helpers import tkns_from_doc


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
            The global configuration for medcat. Usuall cdb.config can be used for this
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
        self.nlp = Pipe(tokenizer=spacy_split_all, config=self.config)
        self.nlp.add_tagger(tagger=partial(tag_skip_and_punct, config=self.config),
                            name='skip_and_punct',
                            additional_fields=['is_punct'])

        spell_checker = BasicSpellChecker(cdb_vocab=self.cdb.vocab, config=self.config, data_vocab=vocab)
        self.nlp.add_token_normalizer(spell_checker=spell_checker, config=self.config)

        # Add NER
        self.ner = NER(self.cdb, self.config)
        self.nlp.add_ner(self.ner)

        # Add LINKER
        self.linker = Linker(self.cdb, vocab, self.config)
        self.nlp.add_linker(self.linker)

        # Add meta_annotaiton classes if they exist
        self._meta_annotations = False
        for meta_cat in meta_cats:
            self.nlp.add_meta_cat(meta_cat, meta_cat.category_name)
            self._meta_annotations = True

        # Set max document length
        self.nlp.nlp.max_length = self.config.preprocessing.get('max_document_length', 1000000)


    def __call__(self, text, do_train=False):
        r'''
        Push the text through the pipeline.

        Args:
            text (string):
                The text to be annotated, if it is longer than self.config.preprocessing['max_document_length'] it will be trimmed
                to that length.
            do_train (bool, defaults to `False`):
                This causes so many screwups when not there, so I'll force training
                to False. To run training it is much better to use the self.train() function
                but for some special cases I'm leaving it here also.
        Returns:
            A spacy document with the extracted entities
        '''
        # Should we train - do not use this for training, unles you know what you are doing. Use the
        #self.train() function
        self.config.linking['train'] = do_train

        if text and len(text) > 0:
            return self.nlp(text[0:self.config.preprocessing.get('max_document_length', 1000000)])
        else:
            return None


    def _print_stats(self, data, epoch=0, use_filters=False, use_overlaps=False, use_cui_doc_limit=False,
                     use_groups=False):
        r''' TODO: Refactor and make nice
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
            examples (dict):
                Examples for each of the fp, fn, tp. Foramt will be examples['fp']['cui'][<list_of_examples>]
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
        _filters = deepcopy(self.config.linking['filters'])

        # Reset all filters
        self.config.linking['filters']['cuis'] = set()

        for pind, project in tqdm(enumerate(data['projects']), desc="Stats project", total=len(data['projects']), leave=False):
            if use_filters:
                self.config.linking['filters']['cuis'] = process_old_project_filters(
                        cuis=project.get('cuis', None), type_ids=project.get('tuis', None), cdb=self.cdb)

            start_time = time.time()
            for dind, doc in tqdm(enumerate(project['documents']), desc='Stats document', total=len(project['documents']), leave=False):
                anns = doc['annotations']

                # Apply document level filtering if
                if use_cui_doc_limit:
                    _cuis = set([ann['cui'] for ann in anns])
                    if _cuis:
                        self.config.linking['filters']['cuis'] = _cuis

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
                    if not use_filters or check_filters(cui, self.config.linking['filters']):
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


        except Exception as e:
            traceback.print_exc()

        self.config.linking['filters'] = _filters

        return fps, fns, tps, cui_prec, cui_rec, cui_f1, cui_counts, examples


    def train(self, data_iterator, fine_tune=True, progress_print=1000):
        """ Runs training on the data, note that the maximum lenght of a line
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
                    self.log.warning("LINE: '{}...' \t WAS SKIPPED".format(line[0:100]))
                    self.log.warning("BECAUSE OF: " + str(e))
                if cnt % progress_print == 0:
                    self.log.info("DONE: " + str(cnt))
                cnt += 1

        self.config.linking['train'] = False


    def add_cui_to_group(self, cui, group_name, reset_all_groups=False):
        r'''
        Ads a CUI to a group, will appear in cdb.addl_info['cui2group']

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
            self.cdb.addl_info['cui2group'] = {}

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
            for name in names:
                cuis.extend(self.cdb.name2cuis.get(name, []))

        # Remove name from all CUIs
        for cui in cuis:
            self.cdb.remove_names(cui=cui, names=names)


    def add_and_train_concept(self, cui, name, spacy_doc=None, spacy_entity=None, ontologies=set(), name_status='A', type_ids=set(),
                              description='', full_build=True, negative=False, devalue_others=False):
        r''' Add a name to an existing concept, or add a new concept, or do not do anything if the name and concept alraedy exist. Perform
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
        self.cdb.add_concept(cui=cui, names=names, ontologies=ontologies, name_status=name_status, type_ids=type_ids, description=description,
                             full_build=full_build)

        if spacy_entity is not None and spacy_doc is not None:
            # Train Linking
            self.linker.context_model.train(cui=cui, entity=spacy_entity, doc=spacy_doc, negative=negative, names=names)

            if not negative and devalue_others:
                # Find all cuis
                cuis = set()
                for name in names:
                    cuis.update(self.cdb.name2cuis.get(name, []))
                # Remove the cui for which we just added positive training
                cuis.remove(cui)
                # Add negative training for all other CUIs that link to these names
                for _cui in cuis:
                    self.linker.context_model.train(cui=_cui, entity=spacy_entity, doc=spacy_doc, negative=True)



    def train_supervised(self, data_path, reset_cui_count=False, nepochs=1,
                         print_stats=True, use_filters=False, terminate_last=False, use_overlaps=False,
                         use_cui_doc_limit=False, test_size=0, devalue_others=False, use_groups=False,
                         never_terminate=False):
        r''' TODO: Refactor, left from old
        Run supervised training on a dataset from MedCATtrainer. Please take care that this is more a simiulated
        online training then supervised.

        Args:
            data_path (str):
                The path to the json file that we get from MedCATtrainer on export.
            reset_cui_count (boolean):
                Used for training with weight_decay (annealing). Each concept has a count that is there
                from the beginning of the CDB, that count is used for annealing. Resetting the count will
                significantly incrase the training impact. This will reset the count only for concepts
                that exist in the the training data.
            nepochs (int):
                Number of epochs for which to run the training.
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
            devalue_others(bool):
                Check add_name for more details.
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

        if reset_cui_count:
            # Get all CUIs
            cuis = []
            for project in train_set['projects']:
                for doc in project['documents']:
                    for ann in doc['annotations']:
                        cuis.append(ann['cui'])
            for cui in set(cuis):
                if cui in self.cdb.cui2count_train:
                    self.cdb.cui2count_train[cui] = 10

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
                            spacy_entity = tkns_from_doc(spacy_doc=spacy_doc, start=start, end=end)
                            deleted = ann.get('deleted', False)
                            self.add_and_train_concept(cui=cui,
                                          name=ann['value'],
                                          spacy_doc=spacy_doc,
                                          spacy_entity=spacy_entity,
                                          negative=deleted,
                                          devalue_others=devalue_others)
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


    def get_entities(self, text, only_cui=False, addl_info=['cui2icd10', 'cui2ontologies']):
        r''' Get entities

        text:  text to be annotated
        return:  entities
        '''
        cnf_annotation_output = getattr(self.config, 'annotation_output', {})
        doc = self(text)
        out = {'entities': {}, 'tokens': []}
        if doc is not None:
            out_ent = {}
            if self.config.general.get('show_nested_entities', False):
                _ents = doc._.ents
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

            for ind, ent in enumerate(_ents):
                cui = str(ent._.cui)
                if not only_cui:
                    out_ent['pretty_name'] = self.cdb.cui2preferred_name.get(cui, '')
                    out_ent['cui'] = cui
                    out_ent['tuis'] = list(self.cdb.cui2type_ids.get(cui, ''))
                    out_ent['types'] = [self.cdb.addl_info['type_id2name'].get(tui, '') for tui in out_ent['tuis']]
                    out_ent['source_value'] = ent.text
                    out_ent['detected_name'] = str(ent._.detected_name)
                    out_ent['acc'] = float(ent._.context_similarity)
                    out_ent['context_similarity'] = float(ent._.context_similarity)
                    out_ent['start'] = ent.start_char
                    out_ent['end'] = ent.end_char
                    for addl in addl_info:
                        tmp = self.cdb.addl_info[addl].get(cui, [])
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
                    out['entities'].append(cui)

        return out


    def get_json(self, text, only_cui=False, addl_info=['cui2icd10', 'cui2ontologies']):
        """ Get output in json format

        text:  text to be annotated
        return:  json with fields {'entities': <>, 'text': text}
        """
        ents = self.get_entities(text, only_cui, addl_info=addl_info)['entities']
        out = {'annotations': ents, 'text': text}

        return json.dumps(out)


    def multiprocessing(self, in_data, nproc=8, batch_size=100, only_cui=False, addl_info=[]):
        r''' Run multiprocessing NOT FOR TRAINING

        in_data:  an iterator or array with format: [(id, text), (id, text), ...]
        nproc:  number of processors
        batch_size:  obvious

        return:  an list of tuples: [(id, doc_json), (id, doc_json), ...]
        '''
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
            p = Process(target=self._mp_cons, kwargs={'in_q': in_q, 'out_dict': out_dict, 'pid': i, 'only_cui': only_cui,
                                                      'addl_info': addl_info})
            p.start()
            procs.append(p)

        data = []
        for id, text in in_data:
            data.append((id, str(text)))
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


    def _mp_cons(self, in_q, out_dict, pid=0, only_cui=False, addl_info=[]):
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
                        doc = self.get_entities(text=text, only_cui=only_cui, addl_info=addl_info)
                        doc['text'] = text
                        out.append((id, doc))
                    except Exception as e:
                        self.log.warning("Exception in _mp_cons")
                        self.log.warning(e, stack_info=True)

            sleep(1)
