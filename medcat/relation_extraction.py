import json
import logging
import os
import numpy
import logging
from numpy.core.fromnumeric import take
import torch

import torch.nn
import pickle
import dill
import torch.optim
import torch
from torch.utils.data import dataloader
import tqdm
import torch.nn as nn
from torch import Tensor
from datetime import date, datetime
from torch.nn.modules.module import T
from transformers import BertConfig
from ast import literal_eval
from itertools import permutations
from pandas.core.series import Series
from medcat.cdb import CDB
from medcat.config_re import ConfigRE
from medcat.utils.relation_extraction.tokenizer import TokenizerWrapperBERT

from spacy.tokens import Doc
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from medcat.utils.meta_cat.ml_utils import split_list_train_test


from medcat.utils.relation_extraction.eval import Two_Headed_Loss
from medcat.utils.relation_extraction.models import BertModel_RelationExtraction
from medcat.utils.relation_extraction.pad_seq import Pad_Sequence

from medcat.utils.relation_extraction.utils import batch_split, create_tokenizer_pretrain, load_bin_file, load_results, load_state, put_blanks, save_bin_file, save_results, tokenize

from medcat.utils.relation_extraction.rel_dataset import RelData
from seqeval.metrics import precision_score, recall_score, f1_score

class RelationExtraction(object):

    name : str = "rel"

    def __init__(self,  cdb : CDB, config: ConfigRE = ConfigRE(), re_model_path : str = "", tokenizer: Optional[TokenizerWrapperBERT] = None, task="train"):
    
       self.config = config
       self.tokenizer = tokenizer
       self.cdb = cdb
      
       self.learning_rate = config.train["lr"]
       self.batch_size = config.train["batch_size"]
       self.n_classes = config.model["nclasses"]

       self.is_cuda_available = torch.cuda.is_available()

       self.device = torch.device("cuda:0" if self.is_cuda_available else "cpu")
       self.hf_model_name = "bert-large-uncased"

       self.model_config = BertConfig.from_pretrained(self.hf_model_name) 

       if self.is_cuda_available:
           self.model = self.model.to(self.device)

       if self.tokenizer is None:
            tokenizer_path = os.path.join(re_model_path, "BERT_tokenizer_relation_extraction")
            if os.path.exists(tokenizer_path):
                self.tokenizer = TokenizerWrapperBERT.load(tokenizer_path)
            else:
                self.tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-large-uncased"),
                                                      max_seq_length=self.model_config.max_position_embeddings) 
                create_tokenizer_pretrain(self.tokenizer)
    
       self.model_config.vocab_size = len(self.tokenizer.hf_tokenizers)

       self.model = BertModel_RelationExtraction.from_pretrained(pretrained_model_name_or_path=self.hf_model_name,
                                                                 model_size=self.hf_model_name,
                                                                 config=self.model_config,
                                                                 task=task,
                                                                 n_classes=self.n_classes)  

       self.model.resize_token_embeddings(self.model_config.vocab_size)
       
       unfrozen_layers = ["classifier", "pooler", "encoder.layer.11", \
                          "classification_layer", "blanks_linear", "lm_linear", "cls"]

       for name, param in self.model.named_parameters():
        if not any([layer in name for layer in unfrozen_layers]):
            param.requires_grad = False
        else:
            param.requires_grad = True
        
       self.pad_id = self.tokenizer.hf_tokenizers.pad_token_id
       self.padding_seq = Pad_Sequence(seq_pad_value=self.pad_id,\
                       label_pad_value=self.pad_id,\
                       label2_pad_value=-1)

    def create_test_train_datasets(self, data):
        train_data, test_data = {}, {}
        train_data["output_relations"], test_data["output_relations"] = split_list_train_test(data["output_relations"],
                                    test_size=self.config.train["test_size"], shuffle=False)
        for k,v in data.items():
            if k != "output_relations":
                train_data[k] = v
                test_data[k] = v

        return train_data, test_data

    def train(self, export_data_path = "", csv_path = "", docs = None, checkpoint_path="./", num_epoch=1, gradient_acc_steps=1, multistep_lr_gamma=0.8, max_grad_norm=1.0):
        
        train_rel_data = RelData(cdb=self.cdb, config=self.config, tokenizer=self.tokenizer)
        test_rel_data = RelData(cdb=CDB(self.cdb.config), config=self.config, tokenizer=None)

        if csv_path != "":
            train_rel_data.dataset, test_rel_data.dataset = self.create_test_train_datasets(train_rel_data.create_base_relations_from_csv(csv_path))
            
            #print(train_rel_data.create_base_relations_from_csv(csv_path))
        elif export_data_path != "":
            export_data = {}
            with open(export_data_path) as f:
                export_data = json.load(f)

            #print(train_rel_data.create_relations_from_export(export_data))
            train_rel_data.dataset, test_rel_data.dataset = self.create_test_train_datasets(train_rel_data.create_relations_from_export(export_data))
        else:
            logging.error("NO DATA HAS BEEN PROVIDED (JSON/CSV/spacy_DOCS)")
            return

        train_dataset_size = len(train_rel_data)
        batch_size = train_dataset_size if train_dataset_size < self.batch_size else self.batch_size
        train_dataloader = DataLoader(train_rel_data, batch_size=batch_size, shuffle=True, \
                                  num_workers=0, collate_fn=self.padding_seq, pin_memory=False)

        test_dataset_size = len(test_rel_data)
        test_batch_size = test_dataset_size if test_dataset_size < self.batch_size else self.batch_size
        test_dataloader = DataLoader(test_rel_data, batch_size=test_batch_size, shuffle=True, \
                                 num_workers=0, collate_fn=self.padding_seq, pin_memory=False)

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.Adam([{"params": self.model.parameters(), "lr": self.learning_rate}])
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,\
                                                                        24,26,30], gamma=multistep_lr_gamma)
        
        start_epoch, best_pred = load_state(self.model, optimizer, scheduler, load_best=False)  

        logging.info("Starting training process...")
        
        losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch = load_results(path=checkpoint_path)

        # update_size = 1 if len(train_dataloader) // 10 > 0

        for epoch in range(start_epoch, num_epoch):
            start_time = datetime.now().time()
            self.model.train()
            # self.model.zero_grad()
            total_loss = 0.0

            losses_per_batch = []
            total_acc = 0.0
            accuracy_per_batch = []
            
            for i, data in enumerate(train_dataloader, 0): 
                token_ids, e1_e2_start, labels, _, _, _ = data
                
                attention_mask = (token_ids != self.pad_id).float()
                token_type_ids = torch.zeros((token_ids.shape[0], token_ids.shape[1])).long()

                classification_logits = self.model(input_ids=token_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask,
                          e1_e2_start=e1_e2_start)
                loss = criterion(classification_logits, labels.squeeze(1))
                loss = loss/gradient_acc_steps

                loss.backward()
            
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                if (i % gradient_acc_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                total_acc += self.evaluate_(classification_logits, labels, ignore_idx=-1)[0]
                
                if (i % batch_size) == (batch_size - 1):
                    losses_per_batch.append(gradient_acc_steps*total_loss/batch_size)
                    accuracy_per_batch.append(total_acc/batch_size)

                    print('[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f' %
                        (epoch + 1, (i + 1)*self.batch_size, train_dataset_size, losses_per_batch[-1], accuracy_per_batch[-1]))
                    total_loss = 0.0; total_acc = 0.0

            end_time = datetime.now().time()
            scheduler.step()
            results = self.evaluate_results(test_dataloader, self.pad_id)
            if len(losses_per_batch) > 0:
                losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
                print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
            if len(accuracy_per_batch) > 0:
                accuracy_per_epoch.append(sum(accuracy_per_batch)/len(accuracy_per_batch))
                print("Train accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))
                 
            test_f1_per_epoch.append(results['f1'])

            print("Epoch finished, took " + str(datetime.combine(date.today(), end_time) - datetime.combine(date.today(), start_time) ) + " seconds")
            print("Test f1 at Epoch %d: %.7f" % (epoch + 1, test_f1_per_epoch[-1]))

            if len(accuracy_per_epoch) > 0 and accuracy_per_epoch[-1] > best_pred:
                best_pred = accuracy_per_epoch[-1]
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': self.model.state_dict(),\
                        'best_acc': accuracy_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./data/" , "training_model_best_BERT.dat"))
            
            if (epoch % 1) == 0:
                save_results(losses_per_epoch, accuracy_per_epoch, test_f1_per_epoch, file_prefix="train")
                #accuracy_per_epoch[-1],\
                torch.save({ 
                        'epoch': epoch + 1,\
                        'state_dict': self.model.state_dict(),
                        'best_acc':  0,  
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict()
                    }, os.path.join("./" , "training_checkpoint_BERT.dat"))

    def evaluate_(self, output, labels, ignore_idx):
        ### ignore index 0 (padding) when calculating accuracy
        idxs = (labels != ignore_idx).squeeze()
        out_labels = torch.softmax(output, dim=1).max(1)[1]
        l = labels.squeeze()[idxs]; 
        o = out_labels[idxs]

        if len(idxs) > 1:
            acc = (l == o).sum().item()/len(idxs)
        else:
            acc = (l == o).sum().item()

        l = l.cpu().numpy().tolist() if l.is_cuda else l.numpy().tolist()
        o = o.cpu().numpy().tolist() if o.is_cuda else o.numpy().tolist()

        return acc, (o, l)

    def evaluate_results(self, dataset, pad_id):
        logging.info("Evaluating test samples...")
        acc = 0
        out_labels = []
        true_labels = []
        self.model.eval()

        with torch.no_grad():
            for i, data in enumerate(dataset):
                logging.info(data)
            
                token_ids, e1_e2_start, labels, _,_,_ = data
                attention_mask = (token_ids != pad_id).float()
                token_type_ids = torch.zeros((token_ids.shape[0], token_ids.shape[1])).long()

                if self.is_cuda_available:
                    token_ids = token_ids.cuda()
                    labels = labels.cuda()
                    attention_mask = attention_mask.cuda()
                    token_type_ids = token_type_ids.cuda()
                    
                classification_logits = self.model(token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                            e1_e2_start=e1_e2_start)
                
                accuracy, (o, l) = self.evaluate_(classification_logits, labels, ignore_idx=-1)

                out_labels.append([str(i) for i in o])
                true_labels.append([str(i) for i in l])
                acc += accuracy
        
        accuracy = acc/(i + 1)
        results = {
            "accuracy": accuracy,
            "precision": precision_score(true_labels, out_labels),
            "recall": recall_score(true_labels, out_labels),
            "f1": f1_score(true_labels, out_labels)
        }

        logging.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logging.info("  %s = %s", key, str(results[key]))
        
        return results
