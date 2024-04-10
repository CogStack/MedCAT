import os
import pickle
from typing import Any, List, Tuple
import numpy as np
import torch

from pandas.core.series import Series
from medcat.config_rel_cat import ConfigRelCAT

from medcat.preprocessing.tokenizers import TokenizerWrapperBERT
from medcat.utils.relation_extraction.models import BertModel_RelationExtraction


def load_bin_file(file_name, path="./") -> Any:
    with open(os.path.join(path, file_name), 'rb') as f:
        data = pickle.load(f)
    return data


def save_bin_file(file_name, data, path="./"):
    with open(os.path.join(path, file_name), "wb") as f:
        pickle.dump(data, f)


def save_state(model: BertModel_RelationExtraction, optimizer=None, scheduler=None, epoch=None, best_f1=None, path="./", model_name="BERT", task="train", is_checkpoint=False, final_export=False):

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


def load_state(model, optimizer, scheduler, path="./", model_name="BERT", file_prefix="train", load_best=False, device=torch.device("cpu"), config: ConfigRelCAT = ConfigRelCAT()):

    model_name = model_name.replace("/", "_")
    print("Attempting to load RelCAT model on device: ", device)
    checkpoint_path = os.path.join(
        path, file_prefix + "_checkpoint_%s.dat" % model_name)
    best_path = os.path.join(
        path, file_prefix + "_best_%s.dat" % model_name)
    start_epoch, best_f1, checkpoint = 0, 0, None

    if load_best is True and os.path.isfile(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        print("Loaded best model.")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Loaded checkpoint model.")

    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(
                [{"params": model.module.parameters(), "lr": config.train.lr}])

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=config.train.multistep_milestones,
                                                             gamma=config.train.multistep_lr_gamma)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Loaded model and optimizer.")

    return start_epoch, best_f1


def save_results(data, model_name="BERT", path="./", file_prefix="train"):
    save_bin_file(file_prefix + "_losses_accuracy_f1_per_epoch_%s.dat" %
                  model_name, data, path)


def load_results(path, model_name="BERT", file_prefix="train"):
    data_dict_path = os.path.join(
        path, file_prefix + "_losses_accuracy_f1_per_epoch_%s.dat" % model_name)

    data_dict = {"losses_per_epoch": [],
                 "accuracy_per_epoch": [], "f1_per_epoch": []}
    if os.path.isfile(data_dict_path):
        data_dict = load_bin_file(data_dict_path)

    return data_dict["losses_per_epoch"], data_dict["accuracy_per_epoch"], data_dict["f1_per_epoch"]


def put_blanks(relation_data: List, blanking_threshold: float = 0.5):
    """
        Args:
            `relation_data` : tuple containing token (sentence_token_span , ent1 , ent2)
        Puts blanks randomly in the relation. Used for pre-training.
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


def create_tokenizer_pretrain(tokenizer, tokenizer_path):
    """
        This method simply adds special tokens that we enouncter 
    """

    tokenizer.hf_tokenizers.add_tokens(
        ["[BLANK]", "[ENT1]", "[ENT2]", "[/ENT1]", "[/ENT2]"], special_tokens=True)
    tokenizer.hf_tokenizers.add_tokens(
        ["[s1]", "[e1]", "[s2]", "[e2]"], special_tokens=True)
    tokenizer.save(tokenizer_path)


# Used for creating data sets for pretraining
def tokenize(relations_dataset: Series, tokenizer: TokenizerWrapperBERT, mask_probability: float = 0.5) -> Tuple:
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
