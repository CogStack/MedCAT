import numpy as np
import json
import copy
from sklearn.metrics import cohen_kappa_score


def count_annotations_project(project):
    cnt = 0
    for doc in project['documents']:
        for ann in doc['annotations']:
            # Only validated
            if ann['validated']:
                cnt += 1
    return cnt


def load_data(data_path, require_annotations=True, order_by_num_ann=True):
    data_candidates = json.load(open(data_path))
    if require_annotations:
        data = {'projects': []}
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
            cnt = count_annotations_project(project)
            cnts.append(cnt)
    srt = np.argsort(-np.array(cnts))
    data['projects'] = [data['projects'][i] for i in srt]

    return data


def count_annotations(data_path):
    data = load_data(data_path, require_annotations=True)

    g_cnt = 0
    for project in data['projects']:
        cnt = count_annotations_project(project)
        g_cnt += cnt
        print("Number of annotations in '{}' is: {}".format(project['name'], cnt))

    print("Total number of annotations is: {}".format(g_cnt))


def get_doc_from_project(project, doc_id):
    for document in project['documents']:
        if document['id'] == doc_id:
            return document
    return None


def get_ann_from_doc(document, start, end):
    for ann in document['annotations']:
        if ann['start'] == start and ann['end'] == end:
            return ann
    return None


def meta_ann_from_ann(ann, meta_name):
    meta_anns = ann['meta_anns']
    # need for old versions of data
    if type(meta_anns) == dict:
        return meta_anns.get(meta_name, None)
    else:
        for meta_ann in meta_anns:
            if meta_ann['name'] == meta_name:
                return meta_ann
    return None


def are_anns_same(ann, ann2, meta_names=[], require_double_inner=True):
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
            elif (meta_ann is None) != (meta_ann2 is None):
                if require_double_inner:
                    return False
    else:
        return False

    return True


def get_same_anns(document, document2, require_double_inner=True, ann_stats=[], meta_names=[]):
    are_same = True
    new_document = copy.deepcopy(document)
    new_document['annotations'] = []

    for ann in document['annotations']:
        # Take only validated anns
        if ann['validated']:
            ann2 = get_ann_from_doc(document2, ann['start'], ann['end'])

            if ann2 is not None:
                if not are_anns_same(ann, ann2, meta_names):
                    ann_stats.append((1, 0))
                    are_same = False
                else:
                    ann_stats.append((1, 1))
                    new_document['annotations'].append(ann)
            elif not require_double_inner:
                ann_stats.append((1, 1))
                new_document['annotations'].append(ann)
            else:
                ann_stats.append((1, 0))
                are_same = False

    # Check the reverse also, but only ann2 if not in first doc
    for ann2 in document2['annotations']:
        ann = get_ann_from_doc(document, ann2['start'], ann2['end'])

        if ann is None:
            if not require_double_inner:
                ann_stats.append((1, 1))
                # Also append this annotation to the document, because it is missing from it
                new_document['annotations'].append(ann2)
            else:
                ann_stats.append((0, 1))
                are_same = False

    return new_document



def print_consolid_stats(ann_stats=[]):
    if ann_stats:
        ann_stats = np.array(ann_stats)
        ck = 0 #TODO: cohen_kappa_score(ann_stats[:, 0], ann_stats[:, 1])
        t = 0
        for i in range(len(ann_stats)):
            if ann_stats[i, 0] == ann_stats[i, 1]:
                t += 1
        agr = t / len(ann_stats)
        print("Kappa: {:.4f}; Agreement: {:.4f}".format(ck, agr)) 
        print("InAgreement vs Total: {} / {}".format(t, len(ann_stats)))


def consolidate_double_annotations(data_path, out_path, require_double=True, require_double_inner=True, meta_anns_to_match=[]):
    """ Consolidated a dataset that was multi-annotated (same documents two times).

    data_path:
        Output from MedCATtrainer - projects containig the same documents must have the same name.
    out_path:
        The consolidated data will be saved here - usually only annotations where both annotators agre
            out_path:
            The consolidated data will be saved here - usually only annotations where both annotators agreee
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
    data = load_data(data_path, require_annotations=True)
    out_data = {'projects': []}
    projects_done = set()
    ann_stats = [] # This will keep score for agreement
    # Consolidate
    for project in data['projects']:
        id = project['id']
        new_documents = []
        ann_stats_project = []
        new_project = None
        if id not in projects_done:
            projects_done.add(id)
            name = project['name']
            documents = project['documents']

            if not require_double:
                new_project = copy.deepcopy(project)
                projects_done.add(id)
            else:
                # Means we need double annotations
                has_double = False
                for project2 in data['projects']:
                    id2 = project2['id']
                    name2 = project2['name']

                    if name == name2 and id != id2:
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
                ann_stats.extend(ann_stats_project)
                out_data['projects'].append(new_project)

            if ann_stats_project:
                print("**Printing stats for project: {}".format(project['name']))
                print_consolid_stats(ann_stats_project)
                print("\n\n")
            else:
                print("**Project '{}' did not have double annotations\n\n".format(project['name']))

    # Save
    json.dump(out_data, open(out_path, 'w'))
    print("**Overall stats")
    print_consolid_stats(ann_stats)

def validate_ner_data(data_path, cdb, cntx_size=70, status_only=False, ignore_if_already_done=False):
    """ Please just ignore this function, I'm afraid to even look at it
    """
    data = json.load(open(data_path))
    name2cui = {}
    cui2status = {}

    print("This will overwrite the original data, make sure you've a backup")
    print("If something is completely wrong or you do not know what to do, chose the [s]kip option, you can also skip by leaving input blank.")
    print("If you want to [q]uit write  q  your progress will be saved to the json and you can continue later")

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

    quit = False

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
                            print("\n\nThis name was annotated with multiple CUIs\n")
                            b = text[max(0, start-cntx_size):start].replace("\n", " ")
                            m = text[start:end].replace("\n", " ")
                            e = text[end:min(len(text), end+cntx_size)].replace("\n", " ")
                            print("SNIPPET: {} <<<{}>>> {}".format(b, m, e))
                            print()
                            print("C | {:3} | {:20} | {:70} | {}".format("ID", "CUI", "Concept", "Number of annotations in the dataset"))
                            print("-"*110)
                            for id, _cui in enumerate(cuis):
                                if _cui == cui:
                                    c = "+"
                                else:
                                    c = " "
                                print("{} | {:3} | {:20} | {:70} | {}".format(c, id, _cui, cdb.cui2pretty_name.get(_cui, 'unk')[:69], name2cui[name][_cui]))
                            print()
                            d = str(input("###Change to ([s]kip/[q]uit/id): "))

                            if d == 'q':
                                quit = True
                                break

                            ann['manul_verification_mention'] = True
                            if d == 's':
                                continue
                            if d.isnumeric():
                                d = int(d)
                                ann['cui'] = cuis[d]

                if quit:
                    break
            if quit:
                break

    if not quit:
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
                            ss = list(cui2status[cui][name].keys())
                            print("\n\nThis name was annotated with different status\n")
                            b = text[max(0, start-cntx_size):start].replace("\n", " ")
                            m = text[start:end].replace("\n", " ")
                            e = text[end:min(len(text), end+cntx_size)].replace("\n", " ")
                            print("SNIPPET           : {} <<<{}>>> {}".format(b, m, e))
                            print("CURRENT STATUS    : {}".format(status))
                            print("CURRENT ANNOTATION: {} - {}".format(cui, cdb.cui2pretty_name.get(cui, 'unk')))
                            print("ANNS TOTAL        :")
                            for k,v in cui2status[cui][name].items():
                                print("        {}: {}".format(str(k), str(v)))
                            print()

                            d = str(input("###Change to ([q]uit/[s]kip/[c]orrect/[i]ncorrect/[t]erminate): "))

                            if d == 'q':
                                quit = True
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
                            print()
                            print()
                if quit:
                    break
            if quit:
                break

    json.dump(data, open(data_path, 'w'))



def prepare_from_json(data, cntx_left, cntx_right, tokenizer, lowercase=True, cntx_in_chars=False, tui_filter=None):
    """ Convert the data from a json format into a CSV-like format for training.

    data:  json file from MedCAT
    cntx_left:  size of the context
    cntx_right:  size of the context
    tokenizer:  instance of the <Tokenizer> class from huggingface

    return:  {'category_name': [('category_value', 'tokens', 'center_token'), ...], ...}
    """
    out_data = {}

    for project in data['projects']:
        for document in project['documents']:
            if lowercase:
                text = str(document['text']).lower()
            else:
                text = str(document['text'])

            if len(text) > 0:
                doc_text = tokenizer.encode(text)

                for ann in document['annotations']:
                    tui = ""
                    if tui_filter:
                        tui = ann['tui']

                    if not tui_filter or tui in tui_filter:
                        if ann.get('validated', True) and (not ann.get('deleted', False) and not ann.get('killed', False)):
                            start = ann['start']
                            end = ann['end']

                            if not cntx_in_chars:
                                # Get the index of the center token
                                ind = 0
                                for ind, pair in enumerate(doc_text.offsets):
                                    if start >= pair[0] and start <= pair[1]:
                                        break

                                _start = max(0, ind - cntx_left)
                                _end = min(len(doc_text.tokens), ind + 1 + cntx_right)
                                tkns = doc_text.tokens[_start:_end]
                                cpos = cntx_left + min(0, ind-cntx_left)
                            else:
                                _start = max(0, start - cntx_left)
                                _end = min(len(text), end + cntx_right)
                                tkns = tokenizer.encode(text[_start:_end]).tokens


                            # Backward compatibility if meta_anns is a list vs dict in the new approach
                            meta_anns = []
                            if 'meta_anns' in ann:
                                meta_anns = ann['meta_anns']

                                if type(meta_anns) == dict:
                                    meta_anns = meta_anns.values()

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


def encode_category_values(data, vals=None):
    data = list(data)
    if vals is None:
        vals = set([x[0] for x in data])
        vals = {name:i for i,name in enumerate(vals)}

    # Map values to numbers
    for i in range(len(data)):
        data[i][0] = vals[data[i][0]]

    return data, vals


def tkns_to_ids(data, tokenizer):
    data = list(data)

    for i in range(len(data)):
        data[i][1] = [tokenizer.token_to_id(tok) for tok in data[i][1]]

    return data


def make_mc_train_test(data, cdb, seed=17, test_size=0.2):
    """ This is a disaster
    """
    cnts = {}
    total_anns = 0
    # Count all CUIs
    for project in data['projects']:
        cui_filter = None
        tui_filter = None

        if 'cuis' in project and len(project['cuis'].strip()) > 0:
            cui_filter = [x.strip() for x in project['cuis'].split(",")]
        if 'tuis' in project and len(project['tuis'].strip()) > 0:
            tui_filter = [x.strip().upper() for x in project['tuis'].split(",")]

        for document in project['documents']:
            for ann in document['annotations']:

                if (cui_filter is None and tui_filter is None) or (cui_filter is not None and ann['cui'] in cui_filter) or \
                   (tui_filter is not None and cdb.cui2tui.get(ann['cui'], 'unk') in tui_filter):
                    if ann['cui'] in cnts:
                        cnts[ann['cui']] += 1
                    else:
                        cnts[ann['cui']] = 1

                    total_anns += 1


    test_cnts = {}
    test_anns = 0
    test_prob = 0.90

    test_set = {'projects': []}
    train_set = {'projects': []}

    for i_project in np.random.permutation(np.arange(0, len(data['projects']))):
        project = data['projects'][i_project]
        cui_filter = None
        tui_filter = None

        test_project = {}
        train_project = {}
        for k, v in project.items():
            if k == 'documents':
                test_project['documents'] = []
                train_project['documents'] = []
            else:
                test_project[k] = v
                train_project[k] = v

        if 'cuis' in project and len(project['cuis'].strip()) > 0:
            cui_filter = [x.strip() for x in project['cuis'].split(",")]
        if 'tuis' in project and len(project['tuis'].strip()) > 0:
            tui_filter = [x.strip().upper() for x in project['tuis'].split(",")]


        for i_document in np.random.permutation(np.arange(0, len(project['documents']))):
            # Do we have enough documents in the test set
            if test_anns / total_anns >= test_size:
                test_prob = 0

            document = project['documents'][i_document]

            # Coutn CUIs for this document
            _cnts = {}
            for ann in document['annotations']:
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
                for ann in document['annotations']:
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
