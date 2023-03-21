import os
import random
import math
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from typing import List, Optional, Tuple, Any, Dict
from torch import nn
from scipy.special import softmax
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBase
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix

import logging


logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_batch_piped_data(data: List, start_ind: int, end_ind: int, device: torch.device, pad_id: int) -> Tuple:
    """Creates a batch given data and start/end that denote batch size, will also add
    padding and move to the right device.

    Args:
        data (List[List[int], int, Optional[int]]):
            Data in the format: [[<[input_ids]>, <cpos>, Optional[int]], ...], the third column is optional
            and represents the output label
        start_ind (int):
            Start index of this batch
        end_ind (int):
            End index of this batch
        device (torch.device):
            Where to move the data
        pad_id (int):
            Padding index

    Returns:
        x ():
            Same as data, but subsetted and as a tensor
        cpos ():
            Center positions for the data
    """
    max_seq_len = max([len(x[0]) for x in data])
    x = [x[0][0:max_seq_len] + [pad_id]*max(0, max_seq_len - len(x[0])) for x in data[start_ind:end_ind]]
    cpos = [x[1] for x in data[start_ind:end_ind]]
    y = None
    if len(data[0]) == 3:
        # Means we have the y column
        y = torch.tensor([x[2] for x in data[start_ind:end_ind]], dtype=torch.long).to(device)

    x = torch.tensor(x, dtype=torch.long).to(device)
    cpos = torch.tensor(cpos, dtype=torch.long).to(device)

    return x, cpos, y


def predict(model: nn.Module, data: List, config: ConfigMetaCAT) -> Tuple:
    """Predict on data used in the meta_cat.pipe

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
    """

    pad_id = config.model['padding_idx']
    batch_size = config.general['batch_size_eval']
    device = config.general['device']
    ignore_cpos = config.model['ignore_cpos']

    model.eval()
    model.to(device)

    num_batches = math.ceil(len(data) / batch_size)
    all_logits = []

    with torch.no_grad():
        for i in range(num_batches):
            x, cpos, _ = create_batch_piped_data(data, i*batch_size, (i+1)*batch_size, device=device, pad_id=pad_id)
            logits = model(x, cpos, ignore_cpos=ignore_cpos)
            all_logits.append(logits.detach().cpu().numpy())

    predictions = []
    confidences = []

    # Can be that there are not logits, data is empty
    if all_logits:
        logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(logits, axis=1)
        confidences = np.max(softmax(logits, axis=1), axis=1)

    return predictions, confidences


def split_list_train_test(data: List, test_size: int, shuffle: bool = True) -> Tuple:
    """Shuffle and randomply split data

    Args:
        data
        test_size
        shuffle
    """
    if shuffle:
        random.shuffle(data)

    test_ind = int(len(data) * test_size)
    test_data = data[:test_ind]
    train_data = data[test_ind:]

    return train_data, test_data


def print_report(epoch: int, running_loss: List, all_logits: List, y: Any, name: str = 'Train') -> None:
    """Prints some basic stats during training

    Args:
        epoch
        running_loss
        all_logits
        y
        name
    """
    if all_logits:
        logger.info('Epoch: %d %s %s', epoch, "*"*50, name)
        logger.info(classification_report(y, np.argmax(np.concatenate(all_logits, axis=0), axis=1)))


def train_model(model: nn.Module, data: List, config: ConfigMetaCAT, save_dir_path: Optional[str] = None) -> Dict:
    """Trains a LSTM model (for now) with autocheckpoints

    Args:
        data
        config
        save_dir_path
    """
    # Get train/test from data
    train_data, test_data = split_list_train_test(data, test_size=config.train['test_size'], shuffle=config.train['shuffle_data'])
    device = torch.device(config.general['device']) # Create a torch device

    class_weights = config.train['class_weights']
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights) # Set the criterion to Cross Entropy Loss
    else:
        criterion = nn.CrossEntropyLoss() # Set the criterion to Cross Entropy Loss
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=config.train['lr'])
    model.to(device) # Move the model to device

    batch_size = config.train['batch_size']
    batch_size_eval = config.general['batch_size_eval']
    pad_id = config.model['padding_idx']
    nepochs = config.train['nepochs']
    ignore_cpos = config.model['ignore_cpos']
    num_batches = math.ceil(len(train_data) / batch_size)
    num_batches_test = math.ceil(len(test_data) / batch_size_eval)

    # Can be pre-calculated for the whole dataset
    y_test = [x[2] for x in test_data]
    y_train = [x[2] for x in train_data]

    winner_report: Dict = {}
    for epoch in range(nepochs):
        running_loss = []
        all_logits = []
        model.train()
        for i in range(num_batches):
            x, cpos, y = create_batch_piped_data(train_data, i*batch_size, (i+1)*batch_size, device=device, pad_id=pad_id)
            logits = model(x, center_positions=cpos, ignore_cpos=ignore_cpos)
            loss = criterion(logits, y)
            loss.backward()
            # Track loss and logits
            running_loss.append(loss.item())
            all_logits.append(logits.detach().cpu().numpy())

            parameters = filter(lambda p: p.requires_grad, model.parameters())
            nn.utils.clip_grad_norm_(parameters, 0.25)
            optimizer.step()

        all_logits_test = []
        running_loss_test = []
        model.eval()

        with torch.no_grad():
            for i in range(num_batches_test):
                x, cpos, y = create_batch_piped_data(test_data, i*batch_size_eval, (i+1)*batch_size_eval, device=device, pad_id=pad_id)
                logits = model(x, cpos, ignore_cpos=ignore_cpos)

                # Track loss and logits
                running_loss_test.append(loss.item())
                all_logits_test.append(logits.detach().cpu().numpy())

        print_report(epoch, running_loss, all_logits, y=y_train, name='Train')
        print_report(epoch, running_loss_test, all_logits_test, y=y_test, name='Test')

        _report = classification_report(y_test, np.argmax(np.concatenate(all_logits_test, axis=0), axis=1), output_dict=True)
        if not winner_report or _report[config.train['metric']['base']][config.train['metric']['score']] > \
                winner_report['report'][config.train['metric']['base']][config.train['metric']['score']]:

            report = classification_report(y_test, np.argmax(np.concatenate(all_logits_test, axis=0), axis=1), output_dict=True)
            winner_report['report'] = report
            winner_report['epoch'] = epoch

            # Save if needed
            if config.train['auto_save_model']:
                if save_dir_path is None:
                    raise Exception(
                        "The `save_dir_path` argument is required if `aut_save_model` is `True` in the config")
                else:
                    path = os.path.join(save_dir_path, 'model.dat')
                    torch.save(model.state_dict(), path)
                    logger.info("\n##### Model saved to %s at epoch: %d and %s/%s: %s #####\n", path, epoch, config.train['metric']['base'],
                          config.train['metric']['score'], winner_report['report'][config.train['metric']['base']][config.train['metric']['score']])

    return winner_report


def eval_model(model: nn.Module, data: List, config: ConfigMetaCAT, tokenizer: TokenizerWrapperBase) -> Dict:
    """Evaluate a trained model on the provided data

    Args:
        model
        data
        config
    """
    device = torch.device(config.general['device']) # Create a torch device
    batch_size_eval = config.general['batch_size_eval']
    pad_id = config.model['padding_idx']
    ignore_cpos = config.model['ignore_cpos']
    class_weights = config.train['class_weights']

    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights) # Set the criterion to Cross Entropy Loss
    else:
        criterion = nn.CrossEntropyLoss() # Set the criterion to Cross Entropy Loss

    y_eval = [x[2] for x in data]
    num_batches = math.ceil(len(data) / batch_size_eval)
    running_loss = []
    all_logits = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i in range(num_batches):
            x, cpos, y = create_batch_piped_data(data, i*batch_size_eval, (i+1)*batch_size_eval, device=device, pad_id=pad_id)
            logits = model(x, cpos, ignore_cpos=ignore_cpos)
            loss = criterion(logits, y)

            # Track loss and logits
            running_loss.append(loss.item())
            all_logits.append(logits.detach().cpu().numpy())

    print_report(0, running_loss, all_logits, y=y_eval, name='Eval')

    score_average = config.train['score_average']
    predictions = np.argmax(np.concatenate(all_logits, axis=0), axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y_eval, predictions, average=score_average)

    labels = [name for (name, _) in sorted(config.general['category_value2id'].items(), key=lambda x:x[1])]
    confusion = pd.DataFrame(
        data=confusion_matrix(y_eval, predictions,),
        columns=["true " + label for label in labels],
        index=["predicted " + label for label in labels],
    )


    examples: Dict = {'FP': {}, 'FN': {}, 'TP': {}}
    id2category_value = {v: k for k, v in config.general['category_value2id'].items()}
    for i, p in enumerate(predictions):
        y = id2category_value[y_eval[i]]
        p = id2category_value[p]
        c = data[i][1]
        tkns = data[i][0]
        assert tokenizer.hf_tokenizers is not None
        text = tokenizer.hf_tokenizers.decode(tkns[0:c]) + " <<"+ tokenizer.hf_tokenizers.decode(tkns[c:c+1]).strip() + ">> " + \
            tokenizer.hf_tokenizers.decode(tkns[c+1:])
        info = "Predicted: {}, True: {}".format(p, y)
        if p != y:
            # We made a mistake
            examples['FN'][y] = examples['FN'].get(y, []) + [(info, text)]
            examples['FP'][p] = examples['FP'].get(p, []) + [(info, text)]
        else:
            examples['TP'][y] = examples['TP'].get(y, []) + [(info, text)]

    return {'precision': precision, 'recall': recall, 'f1': f1, 'examples': examples, 'confusion matrix': confusion}
