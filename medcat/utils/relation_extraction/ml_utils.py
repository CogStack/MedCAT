import torch
import logging
import os
import pickle
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import random

from pandas.core.series import Series
from medcat.config_rel_cat import ConfigRelCAT

from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper
from medcat.utils.relation_extraction.models import Base_RelationExtraction

from torch import nn


logger = logging.getLogger(__name__)


def split_list_train_test_by_class(data: List, sample_limit: int = -1, test_size: float = 0.2, shuffle: bool = True) -> Tuple[List, List]:
    """

    Args:
        data (List): "output_relations": relation_instances, <-- see create_base_relations_from_doc/csv
                    for data columns
        sample_limit (int): limit the number of samples per class, useful for dataset balancing . Defaults to -1.
        test_size (float): Defaults to 0.2.
        shuffle (bool): shuffle data randomly. Defaults to True.

    Returns:
        Tuple[List, List]: train and test datasets
    """

    train_data = []
    test_data = []

    row_id_labels = {row_idx: data[row_idx][5] for row_idx in range(len(data))}
    lbl_id_to_name = {data[row_idx][5]: data[row_idx][4] for row_idx in range((len(data)))}

    count_per_label = {lbl: list(row_id_labels.values()).count(
        lbl) for lbl in set(row_id_labels.values())}

    new_label_count_train = {}
    new_label_count_test = {}

    for lbl_id, count in count_per_label.items():
        if sample_limit != -1 and count > sample_limit:
            count = sample_limit

        _test_records_size = int(count * test_size)

        test_sample_count = 0
        train_sample_count = 0

        if _test_records_size not in [0, 1]:
            for row_idx, _lbl_id in row_id_labels.items():
                if _lbl_id == lbl_id:
                    if test_sample_count < _test_records_size:
                        test_data.append(data[row_idx])
                        test_sample_count += 1
                    else:
                        if sample_limit != -1:
                            if train_sample_count < sample_limit:
                                train_data.append(data[row_idx])
                                train_sample_count += 1
                        else:
                            train_data.append(data[row_idx])
                            train_sample_count += 1

        else:
            for row_idx, _lbl_id in row_id_labels.items():
                if _lbl_id == lbl_id:
                    train_data.append(data[row_idx])
                    test_data.append(data[row_idx])
                    train_sample_count += 1
                    test_sample_count += 1

        new_label_count_test[lbl_id] = test_sample_count
        new_label_count_train[lbl_id] = train_sample_count

    logging.info("Relations after train, test split :  train - " + str(sum(new_label_count_train.values()))  + " | test - " + str(sum(new_label_count_test.values())))

    for label_id in list(lbl_id_to_name.keys()):
        logging.info(" label: " + lbl_id_to_name[label_id] + " samples | train " + str(new_label_count_train[label_id]) + " | test " + str(new_label_count_test[label_id]))

    if shuffle:
        random.shuffle(train_data)
        random.shuffle(test_data)

    return train_data, test_data


def load_bin_file(file_name, path="./") -> Any:
    with open(os.path.join(path, file_name), 'rb') as f:
        data = pickle.load(f)
    return data


def save_bin_file(file_name, data, path="./"):
    with open(os.path.join(path, file_name), "wb") as f:
        pickle.dump(data, f)


def save_state(model: Base_RelationExtraction, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.MultiStepLR, epoch:int = 1, best_f1:float = 0.0, path:str = "./", model_name: str = "BERT", task:str = "train", is_checkpoint=False, final_export=False) -> None:
    """ Used by RelCAT.save() and RelCAT.train()
        Saves the RelCAT model state.
        For checkpointing multiple files are created, best_f1, loss etc. score.
        If you want to export the model after training set final_export=True and leave is_checkpoint=False.

    Args:
        model (Base_RelationExtraction): BertModel_RelationExtraction | LlamaModel_RelationExtraction
        optimizer (torch.optim.AdamW, optional): Defaults to None.
        scheduler (torch.optim.lr_scheduler.MultiStepLR, optional): Defaults to None.
        epoch (int): Defaults to None.
        best_f1 (float): Defaults to None.
        path (str):Defaults to "./".
        model_name (str): . Defaults to "BERT". This is used to checkpointing only.
        task (str): Defaults to "train". This is used to checkpointing only.
        is_checkpoint (bool): Defaults to False.
        final_export (bool): Defaults to False, if True then is_checkpoint must be False also. Exports model.state_dict(), out into"model.dat".
    """

    model_name = model_name.replace("/", "_")
    file_name = "%s_checkpoint_%s.dat" % (task, model_name)

    if not is_checkpoint:
        file_name = "%s_best_%s.dat" % (task, model_name)
        if final_export:
            file_name = "model.dat"
            torch.save(model.state_dict(), os.path.join(path, file_name))

    if is_checkpoint:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_f1':  best_f1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, os.path.join(path, file_name))


def load_state(model: Base_RelationExtraction, optimizer, scheduler, path="./", model_name="BERT", file_prefix="train", load_best=False, device: torch.device =torch.device("cpu"), config: ConfigRelCAT = ConfigRelCAT()) -> Tuple[int, int]:
    """ Used by RelCAT.load() and RelCAT.train()

    Args:
        model (Base_RelationExtraction): BertModel_RelationExtraction | LlamaModel_RelationExtraction, it has to be initialized before calling this method via (Bert/Llama)Model_RelationExtraction(...)
        optimizer (_type_): optimizer
        scheduler (_type_): scheduler
        path (str, optional): Defaults to "./".
        model_name (str, optional): Defaults to "BERT".
        file_prefix (str, optional): Defaults to "train".
        load_best (bool, optional): Defaults to False.
        device (torch.device, optional): Defaults to torch.device("cpu").
        config (ConfigRelCAT): Defaults to ConfigRelCAT().

    Returns:
        Tuple (int, int): last epoch and f1 score.
    """

    model_name = model_name.replace("/", "_")
    logging.info("Attempting to load RelCAT model on device: " + str(device))
    checkpoint_path = os.path.join(
        path, file_prefix + "_checkpoint_%s.dat" % model_name)
    best_path = os.path.join(
        path, file_prefix + "_best_%s.dat" % model_name)
    start_epoch, best_f1, checkpoint = 0, 0, None

    if load_best is True and os.path.isfile(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        logging.info("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        logging.info("Loaded checkpoint model.")

    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)

        if optimizer is None:
            parameters = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.AdamW(params=parameters, lr=config.train.lr, weight_decay=config.train.adam_weight_decay,
                                betas=config.train.adam_betas, eps=config.train.adam_epsilon)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=config.train.multistep_milestones,
                                                             gamma=config.train.multistep_lr_gamma)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logging.info("Loaded model and optimizer.")

    return start_epoch, best_f1


def save_results(data, model_name: str = "BERT", path: str = "./", file_prefix: str = "train"):
    save_bin_file(file_prefix + "_losses_accuracy_f1_per_epoch_%s.dat" %
                  model_name, data, path)


def load_results(path, model_name: str = "BERT", file_prefix: str = "train") -> Tuple[List, List, List]:
    data_dict_path = os.path.join(
        path, file_prefix + "_losses_accuracy_f1_per_epoch_%s.dat" % model_name)

    data_dict: Dict = {"losses_per_epoch": [],
                       "accuracy_per_epoch": [], "f1_per_epoch": []}
    if os.path.isfile(data_dict_path):
        data_dict = load_bin_file(data_dict_path)

    return data_dict["losses_per_epoch"], data_dict["accuracy_per_epoch"], data_dict["f1_per_epoch"]


def put_blanks(relation_data: List, blanking_threshold: float = 0.5) -> List:
    """
    Args:
        relation_data (List): tuple containing token (sentence_token_span , ent1 , ent2)
                                Puts blanks randomly in the relation. Used for pre-training.
        blanking_threshold (float): % threshold to blank token ids. Defaults to 0.5.

    Returns:
        List: data
    """

    blank_ent1 = np.random.uniform()
    blank_ent2 = np.random.uniform()

    blanked_relation = relation_data

    sentence_token_span, ent1, ent2, label, label_id, ent1_types, ent2_types, ent1_id, ent2_id, ent1_cui, ent2_cui, doc_id = (
        *relation_data, )

    if blank_ent1 >= blanking_threshold:
        blanked_relation = [sentence_token_span, "[BLANK]", ent2, label, label_id,
                            ent1_types, ent2_types, ent1_id, ent2_id, ent1_cui, ent2_cui, doc_id]

    if blank_ent2 >= blanking_threshold:
        blanked_relation = [sentence_token_span, ent1, "[BLANK]", label, label_id,
                            ent1_types, ent2_types, ent1_id, ent2_id, ent1_cui, ent2_cui, doc_id]

    return blanked_relation


def create_tokenizer_pretrain(tokenizer: Union[BaseTokenizerWrapper], tokenizer_path: str):
    """ 
        This method simply adds the default special tokens that we ecounter.

    Args:
        tokenizer (BaseTokenizerWrapper): BERT/Llama tokenizer.
        tokenizer_path (str): path where tokenizer is to be saved.
    """

    tokenizer.hf_tokenizers.add_tokens(
        ["[BLANK]", "[ENT1]", "[ENT2]", "[/ENT1]", "[/ENT2]"], special_tokens=True)
    tokenizer.hf_tokenizers.add_tokens(
        ["[s1]", "[e1]", "[s2]", "[e2]"], special_tokens=True)
    tokenizer.save(tokenizer_path)


# Used for creating data sets for pretraining
def tokenize(relations_dataset: Series, tokenizer: Union[BaseTokenizerWrapper], mask_probability: float = 0.5) -> Tuple:
    (tokens, span_1_pos, span_2_pos), ent1_text, ent2_text, label, label_id, ent1_types, ent2_types, ent1_id, ent2_id, ent1_cui, ent2_cui, doc_id = relations_dataset

    cls_token = tokenizer.hf_tokenizers.cls_token
    sep_token = tokenizer.hf_tokenizers.sep_token

    tokens = [token.lower() for token in tokens if tokens != '[BLANK]']

    forbidden_indices = [i for i in range(
        span_1_pos[0], span_1_pos[1])] + [i for i in range(span_2_pos[0], span_2_pos[1])]

    pool_indices = [i for i in range(
        len(tokens)) if i not in forbidden_indices]

    masked_indices = np.random.choice(pool_indices,
                                      size=round(mask_probability *
                                                 len(pool_indices)),
                                      replace=False)

    masked_for_pred = [token.lower() for idx, token in enumerate(
        tokens) if (idx in masked_indices)]

    tokens = [token if (idx not in masked_indices)
              else tokenizer.hf_tokenizers.mask_token for idx, token in enumerate(tokens)]

    if (ent1_text == "[BLANK]") and (ent2_text != "[BLANK]"):
        tokens = [cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]", "[BLANK]", "[/ENT1]"] + \
            tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]"] + tokens[span_2_pos[0]:span_2_pos[1]] + ["[/ENT2]"] + tokens[span_2_pos[1]:] + [sep_token]

    elif (ent1_text == "[BLANK]") and (ent2_text == "[BLANK]"):
        tokens = [cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]", "[BLANK]", "[/ENT1]"] + \
            tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]", "[BLANK]",
                                                   "[/ENT2]"] + tokens[span_2_pos[1]:] + [sep_token]

    elif (ent1_text != "[BLANK]") and (ent2_text == "[BLANK]"):
        tokens = [cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]"] + tokens[span_1_pos[0]:span_1_pos[1]] + ["[/ENT1]"] + \
            tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]", "[BLANK]",
                                                   "[/ENT2]"] + tokens[span_2_pos[1]:] + [sep_token]

    elif (ent1_text != "[BLANK]") and (ent2_text != "[BLANK]"):
        tokens = [cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]"] + tokens[span_1_pos[0]:span_1_pos[1]] + ["[/ENT1]"] + \
            tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]"] + tokens[span_2_pos[0]:span_2_pos[1]] + ["[/ENT2]"] + tokens[span_2_pos[1]:] + [sep_token]

    ent1_ent2_start = ([i for i, e in enumerate(tokens) if e == "[ENT1]"][0], [
                       i for i, e in enumerate(tokens) if e == "[ENT2]"][0])

    token_ids = tokenizer.hf_tokenizers.convert_tokens_to_ids(tokens)
    masked_for_pred = tokenizer.hf_tokenizers.convert_tokens_to_ids(
        masked_for_pred)

    return token_ids, masked_for_pred, ent1_ent2_start


def create_dense_layers(relcat_config: ConfigRelCAT):

    # dense layers
    fc1 = nn.Linear(relcat_config.model.model_size, relcat_config.model.hidden_size)
    fc2 = nn.Linear(relcat_config.model.hidden_size, int(relcat_config.model.hidden_size / 2))
    fc3 = nn.Linear(int(relcat_config.model.hidden_size / 2), relcat_config.train.nclasses)

    return fc1, fc2, fc3


def get_annotation_schema_tag(sequence_output: torch.Tensor, input_ids: torch.Tensor, special_tag: List) -> torch.Tensor:
    """ Gets to token sequences from the sequence_ouput for the specific token 
        tag ids in self.relcat_config.general.annotation_schema_tag_ids.

    Args:
        sequence_output (torch.Tensor): hidden states/embeddings for each token in the input text
        input_ids (torch.Tensor): input token ids
        special_tag (List): special annotation token id pairs

    Returns:
        torch.Tensor: new seq_tags
    """

    idx_start = torch.where(input_ids == special_tag[0]) # returns: row ids, idx of token[0]/star token in row
    idx_end = torch.where(input_ids == special_tag[1]) # returns: row ids, idx of token[1]/end token in row

    seen = [] # List to store seen elements and their indices
    duplicate_indices = []

    for i in range(len(idx_start[0])):
        if idx_start[0][i] in seen:
            duplicate_indices.append(i)
        else:
            seen.append(idx_start[0][i])

    if len(duplicate_indices) > 0:
        logger.info("Duplicate entities found, removing them...")
        for idx_remove in duplicate_indices:
            idx_start_0 = torch.cat((idx_start[0][:idx_remove], idx_start[0][idx_remove + 1:]))
            idx_start_1 = torch.cat((idx_start[1][:idx_remove], idx_start[1][idx_remove + 1:]))
            idx_start = (idx_start_0, idx_start_1) # type: ignore

    seen = []
    duplicate_indices = []

    for i in range(len(idx_end[0])):
        if idx_end[0][i] in seen:
            duplicate_indices.append(i)
        else:
            seen.append(idx_end[0][i])

    if len(duplicate_indices) > 0:
        logger.info("Duplicate entities found, removing them...")
        for idx_remove in duplicate_indices:
            idx_end_0 = torch.cat((idx_end[0][:idx_remove], idx_end[0][idx_remove + 1:]))
            idx_end_1 = torch.cat((idx_end[1][:idx_remove], idx_end[1][idx_remove + 1:]))
            idx_end = (idx_end_0, idx_end_1) # type: ignore

    assert len(idx_start[0]) == input_ids.shape[0]
    assert len(idx_start[0]) == len(idx_end[0])
    sequence_output_entities = []

    for i in range(len(idx_start[0])):
        to_append = sequence_output[i, idx_start[1][i] + 1:idx_end[1][i], ]

        # to_append = torch.sum(to_append, dim=0)
        to_append, _ = torch.max(to_append, axis=0) # type: ignore

        sequence_output_entities.append(to_append)
    sequence_output_entities = torch.stack(sequence_output_entities)

    return sequence_output_entities
