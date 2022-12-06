import json
import torch
import copy
import numpy as np
from sklearn.metrics import cohen_kappa_score
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from medcat.cdb import CDB
from collections import defaultdict
import random

import logging

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_annotations_project(project: Dict, cnt_per_cui=None) -> Tuple[int, Any]:
    cnt = 0
    if cnt_per_cui is None:
        cnt_per_cui = defaultdict(int)
    for doc in project['documents']:
        for ann in doc['annotations']:
            # Only validated
            if ann.get('validated', True):
                cnt += 1
                cnt_per_cui[ann['cui']] += 1
    return cnt, cnt_per_cui


def load_data(data_path: str, require_annotations: bool = True, order_by_num_ann: bool = True) -> Dict:
    """Args:
        require_annotations:
            This will require anns but on project level, any doc in a project needs anns.
    """
    data_candidates = json.load(open(data_path))
    if require_annotations:
        data: Dict = {'projects': []}
        # Keep only projects that have annotations
        for project in data_candidates['projects']:
            keep = False
            for document in project['documents']:
                if len(document['annotations']) > 0:
                    keep = True
                    break
            if keep:
                data['projects'].append(project)
    else:
        data = data_candidates


    cnts = []
    if order_by_num_ann:
        for project in data['projects']:
            cnt, _ = count_annotations_project(project)
            cnts.append(cnt)
        srt = np.argsort(-np.array(cnts))
        data['projects'] = [data['projects'][i] for i in srt]

    return data


def count_annotations(data_path: str) -> Dict:
    data = load_data(data_path, require_annotations=True)

    g_cnt = 0
    cnt_per_cui: Dict = defaultdict(int)
    for project in data['projects']:
        cnt, cnt_per_cui = count_annotations_project(project, cnt_per_cui)
        g_cnt += cnt
        logger.info("Number of annotations in project '%s' is: %d", project['name'], cnt)

    # Count annotates per entity
    logger.info("Total number of annotations is: %d", g_cnt)

    # Annotates per CUI
    cnt_per_cui = dict(cnt_per_cui)
    logger.info("Annotations per CUI: ")
    for row in sorted(cnt_per_cui.items(), key=lambda x: x[1], reverse=True):
        logger.info(row)

    return cnt_per_cui


def get_doc_from_project(project: Dict, doc_id: str) -> Optional[Dict]:
    for document in project['documents']:
        if document['id'] == doc_id:
            return document
    return None


def get_ann_from_doc(document: Dict, start: int, end: int) -> Optional[Dict]:
    for ann in document['annotations']:
        if ann['start'] == start and ann['end'] == end:
            return ann
    return None


def meta_ann_from_ann(ann: Dict, meta_name: Union[Dict, List]) -> Optional[Dict]:
    meta_anns = ann['meta_anns']
    # need for old versions of data
    if type(meta_anns) == dict:
        return meta_anns.get(meta_name, None)
    else:
        for meta_ann in meta_anns:
            if meta_ann['name'] == meta_name:
                return meta_ann
    return None


def are_anns_same(ann: Dict, ann2: Dict, meta_names: List = [], require_double_inner: bool = True) -> bool:
    if ann['cui'] == ann2['cui'] and \
       ann['correct'] == ann2['correct'] and \
       ann['deleted'] == ann2['deleted'] and \
       ann['alternative'] == ann2['alternative'] and \
       ann['killed'] == ann2['killed'] and \
       ann['manually_created'] == ann2['manually_created'] and \
       ann['validated'] == ann2['validated']:
        #Check are meta anns the same if exist
        for meta_name in meta_names:
            meta_ann = meta_ann_from_ann(ann, meta_name)
            meta_ann2 = meta_ann_from_ann(ann2, meta_name)
            if meta_ann is not None and meta_ann2 is not None:
                if meta_ann['value'] != meta_ann2['value']:
                    return False
            elif require_double_inner:
                # Remove all annotations that do not have the required meta_anns, this
                #will basically remove double_anns on the meta_level that are marked as incorrect. 
                #We want this, so all good.
                return False
    else:
        return False

    return True


def get_same_anns(
    document: Dict, document2: Dict, require_double_inner: bool = True, ann_stats: List = [], meta_names: List = []
) -> Dict:
    new_document = copy.deepcopy(document)
    new_document['annotations'] = []

    if not ann_stats:
        ann_stats.append([])
        ann_stats.append([])

        for meta_name in meta_names:
            ann_stats.append([])

    # Add some stats
    for ann in document['annotations']:
        # Take only validated anns
        if ann['validated']:
            ann2 = get_ann_from_doc(document2, ann['start'], ann['end'])
            pair = [0, 0]
            if ann['correct']:
                pair[0] = 1

            if ann2 is not None:
                # Only do meta_anns if both anns exist
                for id_meta, meta_name in enumerate(meta_names):
                    ann_stats[id_meta+2].append(['unk', 'unk'])

                    # For ann1
                    meta_ann = meta_ann_from_ann(ann, meta_name)
                    if meta_ann is not None:
                        ann_stats[id_meta+2][-1][0] = meta_ann['value']
                    # For ann2
                    meta_ann = meta_ann_from_ann(ann2, meta_name)
                    if meta_ann is not None:
                        ann_stats[id_meta+2][-1][1] = meta_ann['value']

                if ann2['correct']:
                    pair[1] = 1

                if not are_anns_same(ann, ann2, meta_names):
                    ann_stats[0].append((1, 0))
                else:
                    ann_stats[0].append((1, 1))
                    new_document['annotations'].append(ann)
            elif not require_double_inner:
                ann_stats[0].append((1, 1))
                new_document['annotations'].append(ann)
            else:
                ann_stats[0].append((1, 0))

            # Append for NER+L stats
            ann_stats[1].append(pair)

    # Check the reverse also, but only ann2 if not in first doc
    for ann2 in document2['annotations']:
        if ann2 is not None:
            ann = get_ann_from_doc(document, ann2['start'], ann2['end'])

        if ann is None and ann2 is not None and ann2['validated']:
            # Add a negative example to the stats
            ann_stats[1].append([0, 1])

            if not require_double_inner:
                ann_stats[0].append((1, 1))
                # Also append this annotation to the document, because it is missing from it
                new_document['annotations'].append(ann2)
            else:
                ann_stats[0].append((0, 1))

    return new_document


def print_consolid_stats(ann_stats: List = [], meta_names: List = []) -> None:
    if ann_stats:
        _ann_stats = np.array(ann_stats[0])
        t = 0
        for i in range(len(_ann_stats)):
            if _ann_stats[i, 0] == _ann_stats[i, 1]:
                t += 1
        logger.info("Overall given the parameters (scores here can be strange, be sure to know what they mean)")
        logger.info("   In Agreement vs Total: %d / %d\n\n", t, len(_ann_stats))

        _ann_stats = np.array(ann_stats[1])
        ck = cohen_kappa_score(_ann_stats[:, 0], _ann_stats[:, 1])
        t = 0
        for i in range(len(_ann_stats)):
            if _ann_stats[i, 0] == _ann_stats[i, 1]:
                t += 1
        agr = t / len(_ann_stats)
        logger.info("NER + L")
        logger.info("   Kappa: %.4f; Agreement: %.4f", ck, agr)
        logger.info("   InAgreement vs Total: %d / %d", t, len(_ann_stats))

        for id_meta, meta_name in enumerate(meta_names):
            if len(ann_stats) > id_meta + 2:
                _ann_stats = np.array(ann_stats[id_meta+2])

                ck = cohen_kappa_score(_ann_stats[:, 0], _ann_stats[:, 1])

                t = 0
                for i in range(len(_ann_stats)):
                    if _ann_stats[i, 0] == _ann_stats[i, 1]:
                        t += 1
                agr = t / len(_ann_stats)
                logger.info("Stats for: %s", meta_name)
                logger.info("   Kappa: %.4f; Agreement: %.4f", ck, agr)
                logger.info("   InAgreement vs Total: %d / %d", t, len(_ann_stats))


# Deprecated and removable?
def check_differences(data_path: str, cat: Any, cntx_size=30, min_acc=0.2, ignore_already_done=False, only_start=False, only_saved=False) -> None:
    data = load_data(data_path, require_annotations=True)
    for pid, project in enumerate(data['projects']):
        logger.info("Starting: %d / %d", pid, len(data['projects']))
        cui_filter = None
        tui_filter = None
        if 'cuis' in project and len(project['cuis'].strip()) > 0:
            cui_filter = set([x.strip() for x in project['cuis'].split(",")])
        if 'tuis' in project and len(project['tuis'].strip()) > 0:
            tui_filter = set([x.strip().upper() for x in project['tuis'].split(",")])
        cat.spacy_cat.TUI_FILTER = tui_filter
        cat.spacy_cat.CUI_FILTER = cui_filter

        cat.spacy_cat.MIN_ACC = -5
        cat.spacy_cat.IS_TRAINER = True
        cat.train = False

        for did, doc in enumerate(project['documents']):
            logger.info("Starting: %d / %d", did, len(project['documents']))
            text = doc['text']

            if not doc.get('_verified', False) or ignore_already_done or only_saved:
                # Get annotations with medcat
                s_doc = cat(text)

                t_anns_norm = []
                p_anns_norm = []
                t_anns_start = []
                p_anns_start = []
                for ann in doc['annotations']:
                    t_anns_norm.append((ann['start'], ann['cui']))
                    t_anns_start.append(ann['start'])
                for ann in s_doc.ents:
                    p_anns_norm.append((ann.start_char, ann._.cui))
                    p_anns_start.append(ann.start_char)

                logger.info("__________________") # TODO - remove?
                logger.info("T: ", t_anns_norm)
                logger.info("P: ", p_anns_norm)


                logger.info("\n\nSTARTING MC TO GT")
                for id_p_ann, p_ann in enumerate(p_anns_norm):
                    if (only_start and p_ann[0] not in t_anns_start) or (not only_start and p_ann not in t_anns_norm):
                        ann = s_doc.ents[id_p_ann]
                        if not only_saved:
                            logger.info("\n\nThis does not exist in gt annotations")
                            start = ann.start_char
                            end = ann.end_char
                            cui = ann._.cui
                            b = text[max(0, start-cntx_size):start].replace("\n", " ").replace('\r', ' ')
                            m = text[start:end].replace("\n", " ").replace('\r', ' ')
                            e = text[end:min(len(text), end+cntx_size)].replace("\n", " ").replace('\r', ' ')
                            logger.info("SNIPPET: %s <<<%s>>> %s", b, m, e)
                            logger.info("%s | %s | %s | %s",
                                cui, cat.cdb.get_name(cui), cat.cdb.cui2tui.get(cui, ''), ann.start_char)
                            logger.info(ann._.acc)
                            d = str(input("###Add as (or empty for skip): 1-Correct, 2-Incorrect, s-save: "))

                            if d:
                                new_ann: Dict = {}
                                new_ann['id'] = 0   # ignore
                                new_ann['user'] = 'auto'
                                new_ann['validated'] = True
                                new_ann['last_modified'] = ''
                                new_ann['manually_created'] = False
                                new_ann['acc'] = ann._.acc
                                new_ann['start'] = ann.start_char
                                new_ann['end'] = ann.end_char
                                new_ann['cui'] = ann._.cui
                                new_ann['value'] = ann.text
                                new_ann['killed'] = False
                                new_ann['alternative'] = False
                                if d == '1':
                                    new_ann['correct'] = True
                                    new_ann['deleted'] = False
                                if d == '2':
                                    new_ann['correct'] = False
                                    new_ann['deleted'] = True
                                if d == 'x':
                                    # Save annotations and return
                                    json.dump(data, open(data_path, 'w'))
                                    return
                                if d == 's':
                                    # Save
                                    new_ann['correct'] = False
                                    new_ann['deleted'] = False
                                    new_ann['_saved'] = True

                                doc['annotations'].append(new_ann)

                logger.info("\n\nSTARTING GT TO MC")
                # Redo
                t_anns_norm = []
                for ann in doc['annotations']:
                    t_anns_norm.append((ann['start'], ann['cui']))
                    t_anns_start.append(ann['start'])
                new_doc_anns = []
                for id_t_ann, t_ann in enumerate(t_anns_norm):
                    add_ann = True
                    ann = doc['annotations'][id_t_ann]
                    if (not only_saved and not only_start and t_ann not in p_anns_norm) or \
                       (not only_saved and only_start and t_ann[0] not in p_anns_start) or \
                       (only_saved and ann.get("_saved", False)):
                        # Check is it correct
                        if not ann.get('_verified', False) or ignore_already_done or (only_saved and ann.get('_saved', False)):
                            logger.info("\n\nThis does not exist in mc annotations or it is a saved item")
                            b = text[max(0, ann['start']-cntx_size):ann['start']].replace("\n", " ").replace('\r', ' ')
                            m = text[ann['start']:ann['end']].replace("\n", " ").replace('\r', ' ')
                            e = text[ann['end']:min(len(text), ann['end']+cntx_size)].replace("\n", " ").replace('\r', ' ')
                            logger.info("SNIPPET: %s <<<%s>>> %s", b, m, e)
                            logger.info("%s | %s | %s", ann['cui'], cat.cdb.get_name(ann['cui']), ann['start'])
                            logger.info("Current status")
                            logger.info("  Correct:     %s", str(ann['correct']))
                            logger.info("  Incorrect:   %s", str(ann['deleted']))
                            logger.info("  Alternative: %s", str(ann['alternative']))
                            logger.info("  Killed:      %s", str(ann['killed']))
                            d = str(input("###Change to (or empty for skip): 1-Correct, 2-Incorrect, d-delete, s-save: "))

                            if d == '1':
                                ann['correct'] = True
                                ann['deleted'] = False
                                ann['killed'] = False
                                ann['alternative'] = False
                            elif d == '2':
                                ann['correct'] = False
                                ann['deleted'] = True
                                ann['killed'] = False
                                ann['alternative'] = False
                            elif d == 'd':
                                add_ann = False
                            elif d == 's':
                                # Save for later
                                ann['_saved'] = True
                            elif d == 'x':
                                # Save annotations and return
                                json.dump(data, open(data_path, 'w'))
                                return
                            ann['_verified'] = True

                            if only_saved and ann.get('_saved', False) and d in ['1', '2']:
                                # Remove if it was saved but now it is done
                                del ann['_saved']
                    if add_ann:
                        new_doc_anns.append(ann)
                doc['annotations'] = new_doc_anns
                doc['_verified'] = True

    json.dump(data, open(data_path, 'w'))


def consolidate_double_annotations(data_path: str, out_path: str, require_double: bool = True, require_double_inner: bool = False, meta_anns_to_match: List = []) -> Dict:
    """Consolidated a dataset that was multi-annotated (same documents two times).

    data_path:
        Output from MedCATtrainer - projects containig the same documents must have the same name.
    out_path:
        The consolidated data will be saved here - usually only annotations where both annotators agree
    require_double (boolean):
        If True everything must be double annotated, meaning there have to be two projects of the same name for each name. Else, it will
            also use projects that do not have double annotiations. If this is False, projects that do not have double anns will be
            included as is, and projects that have will still be checked.
    require_double_inner (boolean):
        If False - this will allow some entities to be annotated by only one annotator and not the other, while still requiring
            annotations to be the same if they exist.
    meta_anns_to_match (boolean):
        List of meta annotations that must match for two annotations to be the same. If empty only the mention
            level will be checked.
    """
    d_stats_proj: Dict = {}
    data: Dict = load_data(data_path, require_annotations=True)
    out_data: Dict = {'projects': []}
    projects_done: Set = set()
    ann_stats: List = [] # This will keep score for agreement
    # Consolidate
    for project in data['projects']:
        id_project = project['id']
        new_documents = []
        ann_stats_project: List = []
        new_project = None
        if id_project not in projects_done:
            projects_done.add(id_project)
            name = project['name']
            documents = project['documents']

            if not require_double:
                new_project = copy.deepcopy(project)
                projects_done.add(id_project)
            else:
                # Means we need double annotations
                has_double = False
                for project2 in data['projects']:
                    id2 = project2['id']
                    name2 = project2['name']

                    if name == name2 and id_project != id2:
                        has_double = True
                        projects_done.add(id2)
                        break

                if has_double:
                    for document in documents:
                        document2 = get_doc_from_project(project2, document['id'])

                        if document2 is not None:
                            new_document = get_same_anns(document, document2, require_double_inner=require_double_inner, ann_stats=ann_stats_project, meta_names=meta_anns_to_match)
                            new_documents.append(new_document)
                        elif not require_double_inner:
                            # Use the base document if we are allowing no double anns
                            new_documents.append(document)

                    new_project = copy.deepcopy(project)
                    new_project['documents'] = new_documents

            if new_project is not None:
                if not ann_stats:
                    for _ in ann_stats_project:
                        ann_stats.append([])
                for irow, one in enumerate(ann_stats_project):
                    ann_stats[irow].extend(one)
                out_data['projects'].append(new_project)

            if ann_stats_project:
                logger.info("** Printing stats for project: %s", project['name'])
                print_consolid_stats(ann_stats_project, meta_names=meta_anns_to_match)
                d_stats_proj[project['name']] = ann_stats_project
                logger.info("\n\n") # TODO - remove?
            else:
                logger.info("** Project '%s' did not have double annotations\n\n", project['name'])

    # Save
    json.dump(out_data, open(out_path, 'w'))
    logger.info("** Overall stats")
    print_consolid_stats(ann_stats, meta_names=meta_anns_to_match)
    return d_stats_proj


def validate_ner_data(data_path: str, cdb: CDB, cntx_size: int = 70, status_only: bool = False, ignore_if_already_done: bool = False) -> None:
    """Please just ignore this function, I'm afraid to even look at it."""
    data: Dict = json.load(open(data_path))
    name2cui: Dict = {}
    cui2status: Dict = {}

    logger.info("This will overwrite the original data, make sure you've a backup")
    logger.info("If something is completely wrong or you do not know what to do, chose the [s]kip option, you can also skip by leaving input blank.")
    logger.info("If you want to [q]uit write  q  your progress will be saved to the json and you can continue later")

    for project in data['projects']:
        for document in project['documents']:
            for ann in document['annotations']:
                name = str(ann['value']).lower()
                cui = ann['cui']
                status = None

                if ann['correct']:
                    status = 'Correct'
                else:
                    status = "Other"

                if name in name2cui:
                    name2cui[name][cui] = name2cui[name].get(cui, 0) + 1
                else:
                    name2cui[name] = {cui: 1}

                if cui in cui2status:
                    if name in cui2status[cui]:
                        cui2status[cui][name][status] = cui2status[cui][name].get(status, 0) + 1
                    else:
                        cui2status[cui][name] = {status: 1}
                else:
                    cui2status[cui] = {name: {status: 1}}

    quit_ = False

    if not status_only:
        for project in data['projects']:
            for document in project['documents']:
                text = str(document['text'])
                for ann in document['annotations']:
                    name = str(ann['value']).lower()
                    cui = ann['cui']
                    status = None
                    start = ann['start']
                    end = ann['end']

                    if 'manul_verification_mention' not in ann or ignore_if_already_done:
                        if ann['correct']:
                            status = 'Correct'
                        else:
                            status = "Other"

                        # First check name
                        if len(name2cui[name]) > 1:
                            cuis = list(name2cui[name].keys())
                            logger.info("\n\nThis name was annotated with multiple CUIs\n")
                            b = text[max(0, start-cntx_size):start].replace("\n", " ")
                            m = text[start:end].replace("\n", " ")
                            e = text[end:min(len(text), end+cntx_size)].replace("\n", " ")
                            logger.info("SNIPPET: %s <<<%s>>> %s", b, m, e)
                            logger.info("C | %3s | %20s | %70s | %s", "ID", "CUI", "Concept", "Number of annotations in the dataset")
                            logger.info("-"*110) # TODO - remove?
                            for id_cui, _cui in enumerate(cuis):
                                if _cui == cui:
                                    c = "+"
                                else:
                                    c = " "
                                logger.info("%s | %3s | %20s | %70s | %s", c, id_cui, _cui, cdb.get_name(_cui)[:69], name2cui[name][_cui])
                            d = str(input("###Change to ([s]kip/[q]uit/id): "))

                            if d == 'q':
                                quit_ = True
                                break

                            ann['manul_verification_mention'] = True
                            if d == 's':
                                continue
                            if d.isnumeric():
                                d_ = int(d)
                                ann['cui'] = cuis[d_]

                if quit_:
                    break
            if quit_:
                break

    if not quit_:
        # Re-calculate
        name2cui = {}
        cui2status = {}
        for project in data['projects']:
            for document in project['documents']:
                for ann in document['annotations']:
                    name = str(ann['value']).lower()
                    cui = ann['cui']
                    status = None

                    if ann['correct']:
                        status = 'Correct'
                    else:
                        status = "Other"

                    if name in name2cui:
                        name2cui[name][cui] = name2cui[name].get(cui, 0) + 1
                    else:
                        name2cui[name] = {cui: 1}

                    if cui in cui2status:
                        if name in cui2status[cui]:
                            cui2status[cui][name][status] = cui2status[cui][name].get(status, 0) + 1
                        else:
                            cui2status[cui][name] = {status: 1}
                    else:
                        cui2status[cui] = {name: {status: 1}}


        for project in data['projects']:
            for document in project['documents']:
                text = str(document['text'])
                for ann in document['annotations']:
                    name = str(ann['value']).lower()
                    cui = ann['cui']
                    status = None
                    start = ann['start']
                    end = ann['end']

                    if 'manual_verification_status' not in ann or ignore_if_already_done:
                        if ann['correct']:
                            status = 'correct'
                        elif ann['deleted']:
                            status = 'incorrect'
                        elif ann['killed']:
                            status = 'terminated'
                        elif ann['alternative']:
                            status = 'alternative'
                        else:
                            status = 'unk'

                        if len(cui2status[cui][name]) > 1:
                            logger.info("\n\nThis name was annotated with different status\n")
                            b = text[max(0, start-cntx_size):start].replace("\n", " ")
                            m = text[start:end].replace("\n", " ")
                            e = text[end:min(len(text), end+cntx_size)].replace("\n", " ")
                            logger.info("SNIPPET           : %s <<<%s>>> %s", b, m, e)
                            logger.info("CURRENT STATUS    : %s", status)
                            logger.info("CURRENT ANNOTATION: %s - %s", cui, cdb.get_name(cui))
                            logger.info("ANNS TOTAL        :")
                            for k,v in cui2status[cui][name].items():
                                logger.info("        %s: %s", str(k), str(v))

                            d = str(input("###Change to ([q]uit/[s]kip/[c]orrect/[i]ncorrect/[t]erminate): "))

                            if d == 'q':
                                quit_ = True
                                break

                            ann['manual_verification_status'] = True
                            if d == 's':
                                continue
                            elif d == 'c':
                                ann['correct'] = True
                                ann['killed'] = False
                                ann['deleted'] = False
                                ann['alternative'] = False
                            elif d == 'i':
                                ann['correct'] = False
                                ann['killed'] = False
                                ann['deleted'] = True
                                ann['alternative'] = False
                            elif d == 't':
                                ann['correct'] = False
                                ann['killed'] = True
                                ann['deleted'] = False
                                ann['alternative'] = False
                if quit_:
                    break
            if quit_:
                break

    json.dump(data, open(data_path, 'w'))


class MetaAnnotationDS(torch.utils.data.Dataset):
    def __init__(self, data: Dict, category_map: Dict):
        """Args:
            data:
                Dictionary of data values.
            category_map:
                Map from category naem to id.
        """
        self.data = data
        self.category_map = category_map

    def __getitem__(self, idx: int) -> Dict:
        item = {}
        for key, value in self.data.items():
            if key != 'labels':
                item[key] = torch.tensor(value[idx])
            else:
                item[key] = torch.tensor(self.category_map[value[idx]])
        return item

    def __len__(self) -> int:
        return len(self.data['input_ids'])


def prepare_from_json_hf(data_path: str,
                         cntx_left: int,
                         cntx_right: int,
                         tokenizer: Any,
                         cui_filter: Optional[Dict] = None,
                         replace_center: Optional[Dict] = None) -> Dict:
    out: Dict = {}
    data: Dict = json.load(open(data_path))

    p_data = prepare_from_json_chars(data, cntx_left=cntx_left, cntx_right=cntx_right, tokenizer=tokenizer,
                               cui_filter=cui_filter, replace_center=replace_center)

    for name in p_data.keys():
        out[name] = {}

        out[name]['labels'] = np.array([x[0] for x in p_data[name]])
        out[name]['input_ids'] = [x[1] for x in p_data[name]]
        out[name]['center_positions'] = np.array([x[2] for x in p_data[name]])
        out[name]['token_type_ids'] = [[0] * len(x) for x in out[name]['input_ids']]
        out[name]['attention_mask'] = [[1] * len(x) for x in out[name]['input_ids']]

    return out


def prepare_from_json_chars(data: Dict,
                            cntx_left: int,
                            cntx_right: int,
                            tokenizer: Any,
                            cui_filter: Optional[Dict] = None,
                            replace_center: Optional[Dict] = None) -> Dict:
    """Convert the data from a json format into a CSV-like format for training.

    Args:
        data (Dict): The json file from MedCAT.
        cntx_left (int): The size of the context.
        cntx_right (int): The size of the context.
        tokenizer (Any): The instance of the <FastTokenizer> class from huggingface.
        cui_filter (Optional[Dict], optional): The CUI filter. Defaults to None.
        replace_center (Optional[Dict], optional): If not None the center word (concept) will be
            replaced with whatever is set. Defaults to None.

    Returns:
        Dict: {'category_name': [('category_value', 'tokens', 'center_token'), ...], ...}
    """
    out_data: Dict = {}

    for project in data['projects']:
        for document in project['documents']:
            text = str(document['text'])

            if len(text) > 0:

                for ann in document["annotations"]:
                    if cui_filter:
                        cui = ann['cui']

                    if cui_filter is None or not cui_filter or cui in cui_filter:
                        if ann.get('validated', True) and (not ann.get('deleted', False) and not ann.get('killed', False)):
                            start = ann['start']
                            end = ann['end']

                            _start = max(0, start - cntx_left)
                            _end = min(len(text), end + cntx_right)
                            t_left = tokenizer(text[_start:start])['input_ids']
                            t_right = tokenizer(text[end:_end])['input_ids']
                            if replace_center is None:
                                t_center = tokenizer(text[start:end])['input_ids']
                            else:
                                t_center = tokenizer(replace_center)['input_ids']

                            tkns = t_left + t_center + t_right
                            cpos = len(t_left)

                            # Backward compatibility if meta_anns is a list vs dict in the new approach
                            meta_anns: Union[List, Dict] = []
                            if 'meta_anns' in ann:
                                meta_anns = ann['meta_anns']

                                if type(meta_anns) == dict:
                                    meta_anns = list(meta_anns.values())

                            # If the annotation is validated
                            for meta_ann in meta_anns:
                                name = meta_ann['name']
                                value = meta_ann['value']

                                sample = [value, tkns, cpos]

                                if name in out_data:
                                    out_data[name].append(sample)
                                else:
                                    out_data[name] = [sample]

    return out_data


def make_mc_train_test(data: Dict, cdb: CDB, test_size: float = 0.2) -> Tuple:
    """This is a disaster."""
    cnts: Dict = {}
    total_anns = 0
    # Count all CUIs
    for project in data['projects']:
        cui_filter = None
        tui_filter = None

        if 'cuis' in project and len(project['cuis'].strip()) > 0:
            cui_filter = [x.strip() for x in project['cuis'].split(",")]

        """
        if 'tuis' in project and len(project['tuis'].strip()) > 0:
            tui_filter = [x.strip().upper() for x in project['tuis'].split(",")]
        """

        for document in project['documents']:
            if type(document['annotations']) == list:
                doc_annotations = document['annotations']
            elif type(document['annotations']) == dict:
                doc_annotations = list(document['annotations'].values())

            for ann in doc_annotations:
                if (cui_filter is None and tui_filter is None) or (cui_filter is not None and ann['cui'] in cui_filter):
                    if ann['cui'] in cnts:
                        cnts[ann['cui']] += 1
                    else:
                        cnts[ann['cui']] = 1

                    total_anns += 1


    test_cnts: Dict = {}
    test_anns = 0
    test_prob = 0.90

    test_set: Dict = {'projects': []}
    train_set: Dict = {'projects': []}

    for i_project in np.random.permutation(np.arange(0, len(data['projects']))):
        project = data['projects'][i_project]
        cui_filter = None
        tui_filter = None

        test_project: Dict = {}
        train_project: Dict = {}
        for k, v in project.items():
            if k == 'documents':
                test_project['documents'] = []
                train_project['documents'] = []
            else:
                test_project[k] = v
                train_project[k] = v

        if 'cuis' in project and len(project['cuis'].strip()) > 0:
            cui_filter = [x.strip() for x in project['cuis'].split(",")]
        """
        if 'tuis' in project and len(project['tuis'].strip()) > 0:
            tui_filter = [x.strip().upper() for x in project['tuis'].split(",")]
        """


        for i_document in np.random.permutation(np.arange(0, len(project['documents']))):
            # Do we have enough documents in the test set
            if test_anns / total_anns >= test_size:
                test_prob = 0

            document = project['documents'][i_document]

            # Coutn CUIs for this document
            _cnts: Dict = {}
            if type(document['annotations']) == list:
                doc_annotations = document['annotations']
            elif type(document['annotations']) == dict:
                doc_annotations = list(document['annotations'].values())

            for ann in doc_annotations:
                if (cui_filter is None and tui_filter is None) or (cui_filter is not None and ann['cui'] in cui_filter) or \
                   (tui_filter is not None and cdb.cui2tui.get(ann['cui'], 'unk') in tui_filter):
                    if ann['cui'] in _cnts:
                        _cnts[ann['cui']] += 1
                    else:
                        _cnts[ann['cui']] = 1


            # Did we get more than 30% of concepts for any CUI with >=10 cnt
            is_test = True
            for cui, v in _cnts.items():
                if (v + test_cnts.get(cui, 0)) / cnts[cui] > 0.3:
                    if cnts[cui] >= 10:
                        # We only care for concepts if count >= 10, else they will be ignored
                        #during the test phase (for all metrics and similar)
                        is_test = False
                        break

            # Add to test set
            if is_test and np.random.rand() < test_prob:
                test_project['documents'].append(document)
                if type(document['annotations']) == list:
                    doc_annotations = document['annotations']
                elif type(document['annotations']) == dict:
                    doc_annotations = list(document['annotations'].values())

                for ann in doc_annotations:
                    if (cui_filter is None and tui_filter is None) or (cui_filter is not None and ann['cui'] in cui_filter) or \
                       (tui_filter is not None and cdb.cui2tui.get(ann['cui'], 'unk') in tui_filter):
                        test_anns += 1
                        if ann['cui'] in test_cnts:
                            test_cnts[ann['cui']] += 1
                        else:
                            test_cnts[ann['cui']] = 1
            else:
                train_project['documents'].append(document)

        test_set['projects'].append(test_project)
        train_set['projects'].append(train_project)

    return train_set, test_set, test_anns, total_anns


def get_false_positives(doc: Dict, spacy_doc: Doc) -> List[Span]:
    if type(doc['annotations']) == list:
        truth = set([(ent['start'], ent['cui']) for ent in doc['annotations']])
    elif type(doc['annotations']) == dict:
        truth = set([(ent['start'], ent['cui']) for ent in doc['annotations'].values()])

    fps = []
    for ent in spacy_doc._.ents:
        if (ent.start_char, ent._.cui) not in truth:
            fps.append(ent)

    return fps
