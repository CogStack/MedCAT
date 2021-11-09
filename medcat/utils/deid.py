import torch
import numpy as np


def deid_document(text, tokenizer, model, cdb, verbose=False):
    label2name = {v:k for k,v in tokenizer.label_map.items()}

    tkns_raw = tokenizer.hf_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = tkns_raw['offset_mapping']
    del tkns_raw['offset_mapping']
    preds = [0] * len(offsets)
    for i in range(((len(tkns_raw['input_ids']) - 1) // 500) + 1):
        tkns = {k:torch.tensor([v[i*500:(i+1)*500]]).to(model.device) for k,v in tkns_raw.items()}
        _preds = model(**tkns).logits[0].detach().cpu().numpy()
        _preds = np.argmax(_preds, axis=1)
        _preds[_preds == 1] = 0 # Ignore the token 1
        preds[i*500:i*500 + len(_preds)] = list(_preds)

    new_text = ""
    start = 0
    end = -1
    p_end = 0
    c_ent = False
    for i in range(len(offsets)):
        if preds[i] != 0:
            if not c_ent:
                start = offsets[i][0]

            # Find index of next token that is not subtoken
            n_real = i
            for k in range(i+1, len(offsets)):
                if tokenizer.id2type[tkns_raw['input_ids'][k]] == 'start':
                    n_real == k
                    break

            if i == len(offsets) -1 or preds[i] != preds[i+1]:
                end = offsets[i][1]
                for j in range(i+1, len(offsets)):
                    if tokenizer.id2type[tkns_raw['input_ids'][j]] == 'start':
                        end = offsets[j-1][1]
                        break
                if verbose:
                    deid = "***{} -> {}***".format(text[start:end], cdb.get_name(label2name[preds[i]]))
                else:
                    deid = "***{}***".format(cdb.get_name(label2name[preds[i]]))
                new_text = new_text + text[p_end:start] + deid
                p_end = end
                c_ent = False
            else:
                c_ent = True
    new_text = new_text + text[p_end:]

    return new_text
