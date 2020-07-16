import numpy as np
import json

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
                            if d == 's':
                                continue

                            if d.isnumeric():
                                d = int(d)
                                ann['cui'] = cuis[d]
                            else:
                                continue

                            ann['manul_verification_mention'] = True
                if quit:
                    break
            if quit:
                break

    if not quit:
        # Re-calculate
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
                        else:
                            status = 'unk'

                        if len(cui2status[cui][name]) > 1:
                            ss = list(cui2status[cui][name].keys())
                            print("This name was annotated with different status\n")
                            b = text[max(0, start-cntx_size):start].replace("\n", " ")
                            m = text[start:end].replace("\n", " ")
                            e = text[end:min(len(text), end+cntx_size)].replace("\n", " ")
                            print("SNIPPET: {} <<<{}>>> {}".format(b, m, e))
                            print("CURRENT STATUS: {}".format(status))
                            print("CURRENT ANNOTATION: {} - {}".format(cui, cdb.cui2pretty_name.get(cui, 'unk')))
                            print()
                            d = str(input("###Change to ([q]uit/[s]kip/[c]orrect/[i]ncorrect/[t]erminate): "))

                            if d == 'q':
                                quit = True
                                break
                            if d == 's':
                                continue
                            elif d == 'c':
                                ann['correct'] = True
                                ann['killed'] = False
                                ann['deleted'] = False
                            elif d == 'i':
                                ann['correct'] = False
                                ann['killed'] = False
                                ann['deleted'] = True
                            elif d == 't':
                                ann['correct'] = False
                                ann['killed'] = True
                                ann['deleted'] = False
                            else:
                                continue
                            print()
                            print()
                            ann['manual_verification_status'] = True
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
