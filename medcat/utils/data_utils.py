def prepare_from_json(data, cntx_size, tokenizer, lowercase=True):
    """ Convert the data from a json format into a CSV-like format for training.

    data:  json file from MedCAT
    cntx_size:  size of the context
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
                    if ann['validated'] and (not ann['deleted'] and not ann['killed']):
                        start = ann['start']
                        end = ann['end']

                        # Get the index of the center token
                        ind = 0
                        for ind, pair in enumerate(doc_text.offsets):
                            if start >= pair[0] and start <= pair[1]:
                                break

                        _start = max(0, ind - cntx_size)
                        _end = min(len(doc_text.tokens), ind + 1 + cntx_size)
                        tkns = doc_text.tokens[_start:_end]
                        cpos = cntx_size + min(0, ind-cntx_size)

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
