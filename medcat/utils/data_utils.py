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
                        if ann['validated'] and (not ann['deleted'] and not ann['killed']):
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

                            # If the annotation is validated
                            for meta_ann in ann['meta_anns']:
                                name = meta_ann['name']
                                value = meta_ann['value']

                                sample = [value, tkns, cpos]

                                if name in out_data:
                                    out_data[name].append(sample)
                                else:
                                    out_data[name] = [sample]

    return out_data


def encode_category_values(data):
    data = list(data)
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
