import torch
import numpy as np
import math
from scipy.special import softmax

def create_batch_piped_data(data, start_ind, end_ind, device, pad_id):
    r''' Creates a batch given data and start/end that denote batch size, will also add
    padding and move to the right device.

    Args:
        data (List[List[List[int], int]]):
            Data in the format: [[<input_ids>, <cpos>], ...]
        start_ind (`int`):
            Start index of this batch
        end_ind (`int`):
            End index of this batch
        device (`torch.device`):
            Where to move the data
        pad_id (`int`):
            Padding index

    Returns:
        x ():
            Same as data, but subsetted and as a tensor
        cpos ():
            Center positions for the data

    '''
    max_seq_len = max([len(x[0]) for x in data])
    x = [x[0][0:max_seq_len] + [pad_id]*max(0, max_seq_len - len(x[0])) for x in data[start_ind:end_ind]]
    cpos = [x[1] for x in data[start_ind:end_ind]]

    x = torch.tensor(x, dtype=torch.long).to(device)
    cpos = torch.tensor(cpos, dtype=torch.long).to(device)

    return x, cpos


def predict(model, data, config):
    r''' Predict on data used in the meta_cat.pipe

    Args:
        data (List[List[List[int], int]]):
            Data in the format: [[<input_ids>, <cpos>], ...]
        config (medcat.config_meta_cat.ConfigMetaCAT):
            Configuration for this meta_cat instance.

    Returns:
        predictions (List[int]):
            For each row of input data a prediction
        confidence (List[float]):
            For each prediction a confidence value
    '''

    pad_id = config.model['padding_idx']
    batch_size = config.general['batch_size_eval']
    device = config.general['device']
    ignore_cpos = config.model['ignore_cpos']

    model.eval()
    model.to(device)

    num_batches = math.ceil(len(data) / batch_size)
    all_logits = []
    for i in range(num_batches):
        x, cpos = create_batch_piped_data(data, i*batch_size, (i+1)*batch_size, device=device, pad_id=pad_id)
        logits = model(x, cpos, ignore_cpos=ignore_cpos)
        all_logits.append(logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    predictions = np.argmax(logits, axis=1)
    confidence = np.max(softmax(logits, axis=1), axis=1)

    return predictions, confidence
