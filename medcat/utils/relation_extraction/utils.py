import os
import pickle
from typing import List, Tuple
import numpy as np

import logging

from pandas.core.series import Series
import torch

from medcat.preprocessing.tokenizers import TokenizerWrapperBERT

def load_bin_file(file_name, path="./")  -> "pickle":
    with open(os.path.join(path, file_name), 'rb') as f:
        data = pickle.load(f)
    return data

def save_bin_file(file_name, data, path="./"):
    with open(os.path.join(path, file_name), "wb") as f:
        pickle.dump(data, f)

def load_state(net, optimizer, scheduler, path="./", model_name="BERT", file_prefix="train", load_best=False):
    """ Loads saved model and optimizer states if exists """
    base_path = path

    checkpoint_path = os.path.join(base_path, file_prefix + "_checkpoint_%s.pth.tar" % model_name)
    best_path = os.path.join(base_path, file_prefix + "_model_best_%s.pth.tar" % model_name)
    start_epoch, best_pred, checkpoint = 0, 0, None
    if (load_best == True) and os.path.isfile(best_path):
        checkpoint = torch.load(best_path)
        logging.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logging.info("Loaded checkpoint model.")
    if checkpoint != None:
        start_epoch = checkpoint['epoch']
        best_pred = checkpoint['best_acc']
        net.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        logging.info("Loaded model and optimizer.")    
    return start_epoch, best_pred


def save_results(losses_per_epoch, accuracy_per_epoch, model_name="BERT", path="./", file_prefix="train"):
    save_bin_file(file_prefix + "_losses_per_epoch_%s.pkl" % model_name, losses_per_epoch, path)
    save_bin_file(file_prefix + "_accuracy_per_epoch__%s.pkl" % model_name, accuracy_per_epoch, path)

def load_results(path, model_name="BERT", file_prefix="train"):
    losses_path = os.path.join(path, file_prefix + "_losses_per_epoch_%s.pkl" % model_name)
    accuracy_path = os.path.join(path, file_prefix + "_accuracy_per_epoch_%s.pkl" % model_name)
    f1_path = os.path.join(path, file_prefix + "_f1_per_epoch_%s.pkl" % model_name)

    losses_per_epoch, accuracy_per_epoch, f1_per_epoch = [], [], []

    if os.path.isfile(losses_path):
        losses_per_epoch = load_bin_file(losses_path)
    if os.path.isfile(accuracy_path):
        accuracy_per_epoch = load_bin_file(accuracy_path)
    if os.path.isfile(f1_path):
        f1_per_epoch = load_bin_file(f1_path)

    return losses_per_epoch, accuracy_per_epoch, f1_per_epoch

def put_blanks(relation_data : List, blanking_threshold : float = 0.5):
    """
        Args:
            `relation_data` : tuple containing token (sentence_token_span , ent1 , ent2)
        Puts blanks randomly in the relation. Used for pre-training.
    """    
    blank_ent1 = np.random.uniform()
    blank_ent2 = np.random.uniform()
    
    blanked_relation = relation_data

    sentence_token_span, ent1, ent2, label, label_id, ent1_types, ent2_types, ent1_id, ent2_id, ent1_cui, ent2_cui, doc_id = (*relation_data, )
    
    if blank_ent1 >= blanking_threshold:
        blanked_relation = [sentence_token_span, "[BLANK]", ent2, label, label_id, ent1_types, ent2_types, ent1_id, ent2_id, ent1_cui, ent2_cui, doc_id]
    
    if blank_ent2 >= blanking_threshold:
        blanked_relation = [sentence_token_span, ent1, "[BLANK]", label, label_id, ent1_types, ent2_types, ent1_id, ent2_id, ent1_cui, ent2_cui, doc_id]
        
    return blanked_relation

def create_tokenizer_pretrain(tokenizer, tokenizer_name="BERT"):
    tokenizer.hf_tokenizers.add_tokens(["[BLANK]", "[ENT1]", "[ENT2]", "[/ENT1]", "[/ENT2]"], special_tokens=True)
    save_bin_file(file_name=tokenizer_name + "_tokenizer_relation_extraction.dat", data=tokenizer)

# Used for creating data sets for pretraining
def tokenize(relations_dataset: Series, tokenizer : TokenizerWrapperBERT, mask_probability : float = 0.5) -> Tuple:

    (tokens, span_1_pos, span_2_pos), ent1_text, ent2_text, label, label_id, ent1_types, ent2_types, ent1_id, ent2_id, ent1_cui, ent2_cui, doc_id = relations_dataset

    cls_token = tokenizer.hf_tokenizers.cls_token
    sep_token = tokenizer.hf_tokenizers.sep_token

    tokens = [token.lower() for token in tokens if tokens != '[BLANK]']

    forbidden_indices = [i for i in range(span_1_pos[0], span_1_pos[1])] + [i for i in range(span_2_pos[0], span_2_pos[1])]

    pool_indices = [ i for i in range(len(tokens)) if i not in forbidden_indices ]

    masked_indices = np.random.choice(pool_indices, \
                                        size=round(mask_probability * len(pool_indices)), \
                                        replace=False)

    masked_for_pred = [token.lower() for idx, token in enumerate(tokens) if (idx in masked_indices)]

    tokens = [token if (idx not in masked_indices) else tokenizer.hf_tokenizers.mask_token for idx, token in enumerate(tokens)]

    if (ent1_text == "[BLANK]") and (ent2_text != "[BLANK]"):
        tokens = [cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]" ,"[BLANK]", "[/ENT1]"] + \
            tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]"] + tokens[span_2_pos[0]:span_2_pos[1]] + ["[/ENT2]"] + tokens[span_2_pos[1]:] + [sep_token]
    
    elif (ent1_text == "[BLANK]") and (ent2_text == "[BLANK]"):
        tokens = [cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]" ,"[BLANK]", "[/ENT1]"] + \
            tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]", "[BLANK]", "[/ENT2]"] + tokens[span_2_pos[1]:] + [sep_token]
    
    elif (ent1_text != "[BLANK]") and (ent2_text == "[BLANK]"):
        tokens = [cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]"] + tokens[span_1_pos[0]:span_1_pos[1]] + ["[/ENT1]"] + \
            tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]", "[BLANK]", "[/ENT2]"] + tokens[span_2_pos[1]:] + [sep_token]
    
    elif (ent1_text != "[BLANK]") and (ent2_text != "[BLANK]"):
        tokens = [cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]"] + tokens[span_1_pos[0]:span_1_pos[1]] + ["[/ENT1]"] + \
            tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]"] + tokens[span_2_pos[0]:span_2_pos[1]] + ["[/ENT2]"] + tokens[span_2_pos[1]:] + [sep_token]

    ent1_ent2_start = ([i for i, e in enumerate(tokens) if e == "[ENT1]"] [0] , [i for i, e in enumerate(tokens) if e == "[ENT2]"] [0])

    token_ids = tokenizer.hf_tokenizers.convert_tokens_to_ids(tokens)
    masked_for_pred = tokenizer.hf_tokenizers.convert_tokens_to_ids(masked_for_pred)

    return token_ids, masked_for_pred, ent1_ent2_start


def batch_split(dataset, chunk_size):
    n = max(1, chunk_size)
    return (dataset[i:i+n] for i in range(0, len(dataset), n))