import torch
import logging
import os
import pickle
from typing import Any, Dict, List, Tuple
import random

from medcat.utils.relation_extraction.tokenizer import BaseTokenizerWrapper_RelationExtraction
from medcat.config_rel_cat import ConfigRelCAT

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


def save_state(model, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.MultiStepLR, epoch:int = 1, best_f1:float = 0.0, path:str = "./", model_name: str = "BERT", task:str = "train", is_checkpoint=False, final_export=False) -> None:
    """ Used by RelCAT.save() and RelCAT.train()
        Saves the RelCAT model state.
        For checkpointing multiple files are created, best_f1, loss etc. score.
        If you want to export the model after training set final_export=True and leave is_checkpoint=False.

    Args:
        model (BaseModel_RelationExtraction): BertModel_RelationExtraction | LlamaModel_RelationExtraction etc.
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


def load_state(model, optimizer, scheduler, path="./", model_name="BERT", file_prefix="train", load_best=False, relcat_config: ConfigRelCAT = ConfigRelCAT()) -> Tuple[int, int]:
    """ Used by RelCAT.load() and RelCAT.train()

    Args:
        model (BaseModel_RelationExtraction): BaseModel_RelationExtraction, it has to be initialized before calling this method via (Bert/Llama)Model_RelationExtraction(...)
        optimizer (_type_): optimizer
        scheduler (_type_): scheduler
        path (str, optional): Defaults to "./".
        model_name (str, optional): Defaults to "BERT".
        file_prefix (str, optional): Defaults to "train".
        load_best (bool, optional): Defaults to False.
        relcat_config (ConfigRelCAT): Defaults to ConfigRelCAT().

    Returns:
        Tuple (int, int): last epoch and f1 score.
    """

    device: torch.device =torch.device(relcat_config.general.device)

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
            optimizer = torch.optim.AdamW(params=parameters, lr=relcat_config.train.lr, weight_decay=relcat_config.train.adam_weight_decay,
                                betas=relcat_config.train.adam_betas, eps=relcat_config.train.adam_epsilon)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=relcat_config.train.multistep_milestones,
                                                             gamma=relcat_config.train.multistep_lr_gamma)
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


def create_tokenizer_pretrain(tokenizer: BaseTokenizerWrapper_RelationExtraction, relcat_config: ConfigRelCAT
                              ) -> BaseTokenizerWrapper_RelationExtraction:
    """ 
        This method simply adds the default special tokens that we ecounter.

    Args:
        tokenizer (BaseTokenizerWrapper_RelationExtraction): BERT/Llama tokenizer.
        relcat_config (ConfigRelCAT): The RelCAT config.

    Returns:
        BaseTokenizerWrapper_RelationExtraction: The same tokenizer.
    """

    tokenizer.hf_tokenizers.add_tokens(relcat_config.general.tokenizer_relation_annotation_special_tokens_tags, special_tokens=True)

    # used in llama tokenizer, may produce issues with other tokenizers
    tokenizer.hf_tokenizers.add_special_tokens(relcat_config.general.tokenizer_other_special_tokens)

    return tokenizer


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
