import json
import logging
import os
import numpy
import torch

import torch.nn
import torch.optim
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import date, datetime
from transformers import BertConfig
from medcat.cdb import CDB
from medcat.config_rel_cat import ConfigRelCAT
from medcat.pipeline.pipe_runner import PipeRunner
from medcat.utils.loggers import add_handlers
from medcat.utils.relation_extraction.tokenizer import TokenizerWrapperBERT

from spacy.tokens import Doc
from typing import Iterable, Iterator, Optional, cast
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from medcat.utils.meta_cat.ml_utils import split_list_train_test

from medcat.utils.relation_extraction.models import BertModel_RelationExtraction
from medcat.utils.relation_extraction.pad_seq import Pad_Sequence

from medcat.utils.relation_extraction.utils import create_tokenizer_pretrain,  load_results, load_state, save_results, save_state

from medcat.utils.relation_extraction.rel_dataset import RelData

class RelCAT(PipeRunner):

    name = "rel_cat"

    log = logging.getLogger(__package__)
    # Add file and console handlers
    log = add_handlers(log)

    def __init__(self, cdb : CDB, config: ConfigRelCAT = ConfigRelCAT(), tokenizer: Optional[TokenizerWrapperBERT] = None, task="train"):
    
        self.config = config
        self.tokenizer = tokenizer
        self.cdb = cdb
      
        self.learning_rate = config.train["lr"]
        self.batch_size = config.train["batch_size"]
        self.nclasses = config.model["nclasses"]

        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_available and self.config.general["device"] != "cpu" else "cpu")
        self.tokenizer = tokenizer
        self.model_config = BertConfig()
        self.model = None
        self.task = task
        self.checkpoint_path = "./"
        self.optimizer = None
        self.scheduler = None
        self.best_f1 = 0.0
        self.epoch = 0

        self.pad_id = self.tokenizer.hf_tokenizers.pad_token_id
        self.padding_seq = Pad_Sequence(seq_pad_value=self.pad_id,\
                       label_pad_value=self.pad_id,\
                       label2_pad_value=-1)

    def save(self, save_path) -> None:
        self.config.save(os.path.join(save_path, "config.json"))
        self.model_config.to_json_file(os.path.join(save_path, "model_config.json"))
        self.tokenizer.save(os.path.join(save_path, self.config.general["tokenizer_name"]))
        
        save_state(self.model, self.optimizer, self.scheduler, self.epoch, self.best_f1,
                    save_path, self.config.general["model_name"],
                    self.task, is_checkpoint=False
                )

    @classmethod
    def load(cls, load_path: str = "./") -> "RelCAT":
        
        cdb = CDB(config=None)
        if os.path.exists(os.path.join(load_path, "cdb.dat")):
            cdb = CDB.load(os.path.join(load_path, "cdb.dat"))
        else:
            print("The default CDB file name 'cdb.dat' doesn't exist in the specified path, you will need to load & set \
                          a CDB manually via rel_cat.cdb = CDB.load('path') ")
        
        config_path = os.path.join(load_path, "config.json")
        config = ConfigRelCAT()
        if os.path.exists(config_path):
            config = cast(ConfigRelCAT, ConfigRelCAT.load(os.path.join(load_path, "config.json")))

        tokenizer = None
        tokenizer_path = os.path.join(load_path, config.general["tokenizer_name"])

        if os.path.exists(tokenizer_path):
            tokenizer = TokenizerWrapperBERT.load(tokenizer_path)
        elif config.general["model_name"]:  
            tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config.general["model_name"]))
            create_tokenizer_pretrain(tokenizer, tokenizer_path)
        else:
            tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased"))
        
        model_config = BertConfig()
        model_config_path = os.path.join(load_path, "model_config.json")

        if os.path.exists(model_config_path):
            print("Loaded config from : ", model_config_path)
            model_config = BertConfig.from_json_file(model_config_path)
        else:
            try:
                model_config = BertConfig.from_pretrained(pretrained_model_name_or_path=config.general["model_name"])
            except:
                print("Config for HF model not found: ", config.general["model_name"], ". Using bert-base-uncased.")
                model_config = BertConfig.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
        
        model_config.vocab_size = len(tokenizer.hf_tokenizers)

        rel_cat = cls(cdb=cdb, config=config, tokenizer=tokenizer, task=config.general["task"])
        rel_cat.model_config = model_config

        device = torch.device("cuda" if  torch.cuda.is_available() and config.general["device"] != "cpu" else "cpu")
        
        try:
            rel_cat.model = BertModel_RelationExtraction.from_pretrained(pretrained_model_name_or_path=config.general["model_name"],
                                                                       model_size=config.general["model_name"],
                                                                       model_config=model_config,
                                                                       task=config.general["task"],
                                                                       nclasses=config.model["nclasses"],
                                                                       ignore_mismatched_sizes=True) 
            print("Loaded HF model : ", config.general["model_name"])
        except:
            print("Failed to load specified HF model, defaulting to 'bert-base-uncased', loading...")
            rel_cat.model = BertModel_RelationExtraction.from_pretrained(pretrained_model_name_or_path="bert-base-uncased",
                                                                        model_size="bert-base-uncased",
                                                                        model_config=model_config,
                                                                        task=config.general["task"],
                                                                        nclasses=config.model["nclasses"],
                                                                        ignore_mismatched_sizes=True) 

        rel_cat.model = rel_cat.model.to(device)

        rel_cat.optimizer = torch.optim.Adam([{"params": rel_cat.model.parameters(), "lr": config.train["lr"]}]) 
        rel_cat.scheduler = torch.optim.lr_scheduler.MultiStepLR(rel_cat.optimizer,
                                                                milestones=config.train["multistep_milestones"],
                                                                gamma=config.train["multistep_lr_gamma"]) 

        rel_cat.epoch, rel_cat.best_f1 = load_state(rel_cat.model, rel_cat.optimizer, rel_cat.scheduler, path=load_path,
                                                    model_name=config.general["model_name"],
                                                    file_prefix=config.general["task"],
                                                    device=device)
        
        return rel_cat

    def create_test_train_datasets(self, data, split_sets=False):
        train_data, test_data = {}, {}
        
        if split_sets:
            train_data["output_relations"], test_data["output_relations"] = split_list_train_test(data["output_relations"],
                            test_size=self.config.train["test_size"], shuffle=False)
        

            test_data_label_names = [rec[4] for rec in test_data["output_relations"]]
            test_data["nclasses"], test_data["unique_labels"], test_data["labels2idx"], test_data["idx2label"] = RelData.get_labels(test_data_label_names, self.config)

            for idx in range(len(test_data["output_relations"])):
                test_data["output_relations"][idx][5] = test_data["labels2idx"][test_data["output_relations"][idx][4]]
        else:
            train_data["output_relations"] = data["output_relations"]

        for k, v in data.items():
            if k != "output_relations":
                train_data[k] = []
                test_data[k] = []

        train_data_label_names = [rec[4] for rec in train_data["output_relations"]]
        train_data["nclasses"], train_data["labels2idx"], train_data["idx2label"] = RelData.get_labels(train_data_label_names, self.config)

        for idx in range(len(train_data["output_relations"])):
            train_data["output_relations"][idx][5] = train_data["labels2idx"][train_data["output_relations"][idx][4]]

        return train_data, test_data

    def train(self, export_data_path = "", train_csv_path = "", test_csv_path = "", checkpoint_path="./"):
        
        if self.is_cuda_available:
           self.model = self.model.to(self.device)
           print("Training on device:", torch.cuda.get_device_name(0), self.device)
        
        self.model_config.vocab_size = len(self.tokenizer.hf_tokenizers)

        train_rel_data = RelData(cdb=self.cdb, config=self.config, tokenizer=self.tokenizer)
        test_rel_data = RelData(cdb=CDB(self.cdb.config), config=self.config, tokenizer=self.tokenizer)

        if train_csv_path != "":
            if test_csv_path != "":
                train_rel_data.dataset, _ = self.create_test_train_datasets(train_rel_data.create_base_relations_from_csv(train_csv_path), split_sets=False)
                test_rel_data.dataset, _ = self.create_test_train_datasets(train_rel_data.create_base_relations_from_csv(test_csv_path), split_sets=False)
            else:
                train_rel_data.dataset, test_rel_data.dataset = self.create_test_train_datasets(train_rel_data.create_base_relations_from_csv(train_csv_path), split_sets=True)
          
        elif export_data_path != "":
            export_data = {}
            with open(export_data_path) as f:
                export_data = json.load(f)
            train_rel_data.dataset, test_rel_data.dataset = self.create_test_train_datasets(train_rel_data.create_relations_from_export(export_data), split_sets=True)
        else:
            raise ValueError("NO DATA HAS BEEN PROVIDED (JSON/CSV/spacy_DOCS)")

        train_dataset_size = len(train_rel_data)
        batch_size = train_dataset_size if train_dataset_size < self.batch_size else self.batch_size
        train_dataloader = DataLoader(train_rel_data, batch_size=batch_size, shuffle=False, \
                                  num_workers=0, collate_fn=self.padding_seq, pin_memory=False)

        test_dataset_size = len(test_rel_data)
        test_batch_size = test_dataset_size if test_dataset_size < self.batch_size else self.batch_size
        test_dataloader = DataLoader(test_rel_data, batch_size=test_batch_size, shuffle=False, \
                                 num_workers=0, collate_fn=self.padding_seq, pin_memory=False)

        criterion = nn.CrossEntropyLoss(ignore_index=-1)    

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam([{"params": self.model.parameters(), "lr": self.learning_rate}])
            
        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.train["multistep_milestones"], gamma=self.config.train["multistep_lr_gamma"])
        
        self.epoch, self.best_f1 = load_state(self.model, self.optimizer, self.scheduler, load_best=False, path=checkpoint_path,device=self.device) 

        self.log.info("Starting training process...")
        
        losses_per_epoch, accuracy_per_epoch, f1_per_epoch = load_results(path=checkpoint_path)

        self.model.nclasses = self.config.model["nclasses"] = train_rel_data.dataset["nclasses"]
        self.config.general["labels2idx"].update(train_rel_data.dataset["labels2idx"])

        gradient_acc_steps = self.config.train["gradient_acc_steps"]
        max_grad_norm = self.config.train["max_grad_norm"]

        for epoch in range(self.epoch, self.config.train["nepochs"]):
            start_time = datetime.now().time()
            self.model.train()
            
            total_loss = 0.0

            loss_per_batch = []
            accuracy_per_batch = []

            self.log.info("epoch %d" % epoch)
            
            for i, data in enumerate(train_dataloader, 0): 

                current_batch_size = len(data[0])

                self.log.info("Processing batch %d of epoch %d , batch: %d / %d" % ( i + 1, epoch, (i + 1) * current_batch_size, train_dataset_size ))
                token_ids, e1_e2_start, labels, _, _, _ = data

                attention_mask = (token_ids != self.pad_id).float().to(self.device)
                token_type_ids = torch.zeros((token_ids.shape[0], token_ids.shape[1])).long().to(self.device)
                labels = labels.to(self.device)

                model_output, classification_logits = self.model(
                            input_ids=token_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            e1_e2_start=e1_e2_start
                          )

                batch_loss = criterion(classification_logits.view(-1, self.model.nclasses).to(self.device), labels.squeeze(1))
                batch_loss = batch_loss / gradient_acc_steps
                
                total_loss += batch_loss.item() / current_batch_size

                batch_loss.backward()

                batch_acc, _, batch_precision, batch_f1, _, _ = self.evaluate_(classification_logits, labels, ignore_idx=-1)                
            
                loss_per_batch.append(batch_loss / current_batch_size)
                accuracy_per_batch.append(batch_acc)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                if (i % gradient_acc_steps) == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                print('[Epoch: %d, %5d/ %d points], loss per batch, accuracy per batch: %.3f, %.3f, average total loss %.3f , total loss %.3f' %
                    (epoch, (i + 1) * current_batch_size, train_dataset_size, loss_per_batch[-1], accuracy_per_batch[-1], total_loss / (i + 1), total_loss))

            if len(loss_per_batch) > 0:
                losses_per_epoch.append(sum(loss_per_batch)/len(loss_per_batch))
                print("Losses at Epoch %d: %.5f" % (epoch, losses_per_epoch[-1]))
                
            if len(accuracy_per_batch) > 0:
                accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
                print("Train accuracy at Epoch %d: %.5f" % (epoch, accuracy_per_epoch[-1]))

            total_loss = total_loss / (i + 1)

            end_time = datetime.now().time()
            self.scheduler.step()

            results = self.evaluate_results(test_dataloader, self.pad_id)
                 
            f1_per_epoch.append(results['f1'])

            print("Test f1 at Epoch %d: %.5f" % (epoch, f1_per_epoch[-1]))
            print("Epoch finished, took " + str(datetime.combine(date.today(), end_time) - datetime.combine(date.today(), start_time) ) + " seconds")

            self.epoch = epoch

            if len(f1_per_epoch) > 0 and f1_per_epoch[-1] > self.best_f1:
                self.best_f1 = f1_per_epoch[-1]
                save_state(self.model, self.optimizer, self.scheduler, epoch, self.best_f1, checkpoint_path,\
                            model_name=self.config.general["model_name"], task=self.task, is_checkpoint=False)
            
            if (epoch % 1) == 0:
                save_results({ "losses_per_epoch": losses_per_epoch, "accuracy_per_epoch": accuracy_per_epoch, "f1_per_epoch" : f1_per_epoch}, file_prefix="train", path=checkpoint_path)
                save_state(self.model, self.optimizer, self.scheduler, epoch, self.best_f1, checkpoint_path,\
                            model_name=self.config.general["model_name"], task=self.task)
                
    def evaluate_(self, output_logits, labels, ignore_idx):
        ### ignore index (padding) when calculating accuracy
        idxs = (labels != ignore_idx).squeeze()
        labels_ = labels.squeeze()[idxs].to(self.device)
        pred_labels = torch.softmax(output_logits, dim=1).max(1)[1]
        pred_labels = pred_labels[idxs].to(self.device)

        size_of_batch = len(idxs)

        if len(idxs) > 1:
            acc = (labels_ == pred_labels).sum().item() / size_of_batch
        else:
            acc = (labels_ == pred_labels).sum().item()

        true_labels = labels_.cpu().numpy().tolist() if labels_.is_cuda else labels_.numpy().tolist()
        pred_labels = pred_labels.cpu().numpy().tolist() if pred_labels.is_cuda else pred_labels.numpy().tolist()

        unique_labels = set(true_labels)

        stat_per_label = dict()
        
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

        for label in unique_labels:
            stat_per_label[label] = {"tp": 0, "fp" : 0, "tn" : 0, "fn" : 0}
            for true_label, pred_label in zip(true_labels, pred_labels):
                if label == true_label and label == pred_label:
                    stat_per_label[label]["tp"] += 1
                    total_tp += 1
                elif label == pred_label and true_label != pred_label:
                    stat_per_label[label]["fp"] += 1
                    total_fp += 1
                if true_label == label and label != pred_label:
                    stat_per_label[label]["tn"] += 1
                    total_tn += 1
                elif true_label != label and pred_label == label:
                    stat_per_label[label]["fn"] += 1
                    total_fn += 1

        tp_fn = total_fn + total_tp
        tp_fn = tp_fn if tp_fn > 0.0 else 1.0
        tp_fp = total_fp + total_tp
        tp_fp = tp_fp if tp_fp > 0.0 else 1.0 

        recall =  total_tp / tp_fn
        precision = total_tp / tp_fp

        re_pr = recall + precision 
        re_pr = re_pr if re_pr > 0.0 else 1.0
        f1 = (2 * (recall * precision)) / re_pr

        return acc, recall, precision, f1, pred_labels, true_labels

    def evaluate_results(self, data_loader, pad_id):
        self.log.info("Evaluating test samples...")
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        total_loss, total_acc, total_f1, total_recall, total_precision = 0.0, 0.0, 0.0, 0.0, 0.0
        all_true_labels = None
        all_pred_labels = None
        pred_logits = None
        
        self.model.eval()

        num_samples = len(data_loader)
      
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                token_ids, e1_e2_start, labels, _, _, _ = data
                attention_mask = (token_ids != pad_id).float()
                token_type_ids = torch.zeros((token_ids.shape[0], token_ids.shape[1])).long()

                labels = labels.to(self.device)

                model_output, pred_classification_logits = self.model(token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                            e1_e2_start=e1_e2_start)

                batch_loss = criterion(pred_classification_logits.view(-1, self.model.nclasses).to(self.device), labels.squeeze(1))
                total_loss += batch_loss.item()

                pred_logits = pred_classification_logits if pred_logits is None \
                     else numpy.append(pred_logits, pred_classification_logits, axis=0)

                batch_accuracy, batch_recall, batch_precision, batch_f1, pred_labels, true_labels = \
                    self.evaluate_(pred_classification_logits, labels, ignore_idx=-1)
                
                pred_labels = torch.tensor(pred_labels)
                true_labels = torch.tensor(true_labels)

                all_true_labels = true_labels if all_true_labels is None else torch.cat((all_true_labels, true_labels))
                all_pred_labels = pred_labels if all_pred_labels is None else torch.cat((all_pred_labels, pred_labels))
                
                all_true_labels = all_true_labels.to(self.device)
                all_pred_labels = all_pred_labels.to(self.device)

                total_acc += batch_accuracy
                total_recall += batch_recall
                total_precision += batch_precision
                total_f1 += batch_f1

        total_loss = total_loss / (i + 1)
        total_acc = total_acc / (i + 1)
        total_precision = total_precision / (i + 1)
        total_f1 = total_f1 / (i + 1)
        total_recall = total_recall / (i + 1)

        results = {
            "loss" : total_loss,
            "accuracy": total_acc,
            "precision": total_precision,
            "recall": total_recall,
            "f1": total_f1
        }

        self.log.info("============= Evaluation Results =============")
        for key in sorted(results.keys()):
            self.log.info("  %s = %s", key, str(results[key]))
        
        return results

    def pipe(self, docs: Iterable[Doc], *args, **kwargs) -> Iterator[Doc]:

        predict_rel_dataset = RelData(cdb=self.cdb, config=self.config, tokenizer=self.tokenizer)
        
        Doc.set_extension("relations", default=[], force=True)

        idx2labels = {v : k for k, v in self.config.general["labels2idx"].items()}
        
        if self.is_cuda_available:
            self.model = self.model.to(self.device)
        
        for doc_id, doc in enumerate(docs, 0):
            predict_rel_dataset.dataset, _ = self.create_test_train_datasets(predict_rel_dataset.create_base_relations_from_doc(doc, doc_id), False)
            
            predict_dataloader = DataLoader(predict_rel_dataset, shuffle=False,  batch_size=10, \
                                  num_workers=0, collate_fn=self.padding_seq, pin_memory=False)
            
            total_rel_found = len(predict_rel_dataset.dataset["output_relations"])
            rel_idx = -1
            print("total relations for doc: ", total_rel_found)
            print("processing...")
            pbar = tqdm(total=total_rel_found)
            for i, data in enumerate(predict_dataloader):
                with torch.no_grad():
                    token_ids, e1_e2_start, labels, _, _, _ = data
                    
                    attention_mask = (token_ids != self.pad_id).float()
                    token_type_ids = torch.zeros(token_ids.shape[0], token_ids.shape[1]).long()

                    model_output, pred_classification_logits = self.model(token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                            e1_e2_start=e1_e2_start)
                    for i, pred_rel_logits in enumerate(pred_classification_logits):
                        rel_idx += 1

                        confidence = torch.softmax(pred_rel_logits, dim=0).max(0)
                        predicted_label_id = confidence[1].item()
                      
                        doc._.relations.append({"relation": idx2labels[predicted_label_id], "label_id" : predicted_label_id,
                                                "ent1_text" : predict_rel_dataset.dataset["output_relations"][rel_idx][2], 
                                                "ent2_text" : predict_rel_dataset.dataset["output_relations"][rel_idx][3],
                                                "confidence" : float("{:.3f}".format(confidence[0])),
                                                "start_ent_pos" : "",
                                                "end_ent_pos" : "",
                                                "start_entity_id" : predict_rel_dataset.dataset["output_relations"][rel_idx][8],
                                                "end_entity_id" :  predict_rel_dataset.dataset["output_relations"][rel_idx][9]})
                    pbar.update(len(token_ids))
            pbar.close()
        
            yield doc

    def __call__(self, doc: Doc) -> Doc:
        doc = next(self.pipe(iter([doc])))
        return doc