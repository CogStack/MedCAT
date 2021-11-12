import logging
import os
import numpy
import spacy
import logging
import pandas
import torch

import torch.nn
import pickle
import dill
import regex as re
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.optim
import torch
import torch.nn as nn
from datetime import date, datetime
from itertools import combinations
from torch.nn.modules.module import T
from torch.nn.utils.rnn import pad_sequence
from transformers import BertConfig

from transformers.configuration_utils import PretrainedConfig
from itertools import permutations
from pandas.core.series import Series

from medcat.preprocessing.tokenizers import TokenizerWrapperBERT

from spacy.tokens import Doc
from typing import Dict, Iterable, List, Set, Tuple
from transformers import AutoTokenizer

from tqdm.autonotebook import tqdm

from medcat.utils.models import BertModel_RelationExtracation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_bin_file(file_name, path="./"):
    with open(os.path.join(path, file_name), 'rb') as f:
        data = pickle.load(f)
    return data

def save_bin_file(file_name, data, path="./"):
    with open(os.path.join(path, file_name), "wb") as f:
        pickle.dump(data, f)

class RelData(object):

    Doc.set_extension("relations", default=[], force=True)
    Doc.set_extension("ents", default=[], force=False)

    def __init__(self, docs):
        self.create_base_relations(docs)
        self.relations_dataframe = pandas.DataFrame(self.get_all_instances(docs), columns=["relation", "ent1", "ent2"])

    def create_base_relations(self, docs):
        for doc_id, doc in docs.items():
            if len(doc._.relations) == 0:
                doc._.ents = doc.ents
                doc._.relations = self.get_instances(doc)

    def get_instances(self, doc, window_size=30):
        """  
            doc : SpacyDoc
            window_size : int, Character distance between any two entities start positions.
            Creates a list of tuples based on pairs of entities detected (relation, ent1, ent2) for one spacy document.
        """
        relation_instances = []
        
        doc_length = len(doc)

        chars_to_exclude = ":!@#$%^&*()-+?_=,<>/"

        j = 1
        for ent1 in doc.ents:
            j += 1
            for ent2 in doc.ents[j:]:
                if ent1.text.lower() != ent2.text.lower() and window_size and 1 < ent2.start - ent1.start <= window_size:
                    is_char_punctuation = False
                    start_pos = ent1.start

                    while not is_char_punctuation and start_pos >= 0 :
                        is_char_punctuation = doc[start_pos].is_punct
                        start_pos -= 1

                    left_sentence_context = start_pos + 1 if start_pos > 0 else 0

                    is_char_punctuation = False
                    start_pos = ent2.end

                    while not is_char_punctuation and start_pos < doc_length:
                        is_char_punctuation = doc[start_pos].is_punct
                        start_pos += 1
                    
                    right_sentence_context= start_pos + 1 if start_pos > 0 else doc_length

                    if window_size < (right_sentence_context - left_sentence_context):
                        continue

                    sentence_window_tokens = []

                    for token in doc[left_sentence_context:right_sentence_context]:
                        text = token.text.strip()
                        if text != "" and not any(chr in chars_to_exclude for chr in text):
                            sentence_window_tokens.append(token.text)

                    sentence_token_span = (sentence_window_tokens,
                                     (ent1.start - left_sentence_context, ent1.end - left_sentence_context ),
                                     (ent2.start - left_sentence_context, ent2.end - left_sentence_context)
                       )

                    relation_instances.append((sentence_token_span, ent1.text, ent2.text))

        return relation_instances

    def get_all_instances(self, docs):
        relation_instances = []
        for doc_id, doc in docs.items():
            relation_instances.extend(doc._.relations)
        return relation_instances
        
    def get_subject_objects(self, entity):
        """
            entity: spacy doc entity
        """
        root = entity.sent.root
        subject = None 
        dependencies = []
        pairs = []

        for child in list(root.children):
            if child.dep_ in ["nsubj", "nsubjpass"]: #, "nmod:poss"]:
                subject = child; 
            elif child.dep_ in ["compound", "dobj", "conj", "attr", "ccomp"]:
                dependencies.append(child)

        if subject is not None and len(dependencies) > 0:
            for a, b in permutations([subject] + dependencies, 2):
                a_ = [w for w in a.subtree]
                b_ = [w for w in b.subtree]
                pairs.append((a_, b_))
                
        return pairs

    def __len__(self):
        return len(self.relations_dataframe)

class RelationExtraction(object):

    name : str = "rel"

    def __init__(self, docs, batch_size=100, config: PretrainedConfig = None, spacy_model : str = None, tokenizer = None, embeddings=None, model_name=None, threshold: float = 0.1):
    
       self.config = config
       self.cfg = {"labels": [], "threshold": threshold }
       self.tokenizer = tokenizer
       self.embeddings = embeddings
       self.learning_rate = 0.1
       self.batch_size = batch_size

       self.is_cuda_available = torch.cuda.is_available()

       if model_name is None or model_name == "BERT":
           self.config = BertConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.2") #BertConfig.from_pretrained("bert-base-uncased")  
           self.model = BertModel_RelationExtracation.from_pretrained(pretrained_model_name_or_path="dmis-lab/biobert-base-cased-v1.2", model_size="dmis-lab/biobert-base-cased-v1.2", config=config)  
        
       if self.is_cuda_available:
           self.model = self.model.to(device)

       if self.tokenizer is None:
            if os.path.isfile("./BERT_tokenizer_relation_extraction.dat"):
                self.tokenizer = load_bin_file(file_name="BERT_tokenizer_relation_extraction.dat")
            else:
                self.tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path="emilyalsentzer/Bio_ClinicalBERT"))

       if self.embeddings is None:
           embeddings = numpy.load(os.path.join("./", "embeddings.npy"), allow_pickle=False)
           self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

       self.spacy_nlp = spacy.load("en_core_sci_md") if spacy_model is None else spacy.load(spacy_model)

       self.rel_data = RelData(docs)

       self.alpha = 0.6
       self.mask_probability = 0.10

    def train(self, num_epoch=15, gradient_acc_steps=4, multistep_lr_gamma=0.8):
        self.model.resize_token_embeddings(len(self.tokenizer.hf_tokenizers))

        criterion = Two_Headed_Loss(lm_ignore_idx=self.tokenizer.hf_tokenizers.pad_token_id, use_logits=True, normalize=False)
        optimizer = torch.optim.Adam([{"params" : self.model.parameters(), "lr": self.learning_rate}])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,16,18,20,22,30], gamma=multistep_lr_gamma)

        start_epoch, best_pred, amp_checkpoint = Two_Headed_Loss.load_state(self.model, optimizer, scheduler, load_best=False)  

        losses_per_epoch, accuracy_per_epoch = Two_Headed_Loss.load_results()
        
        logging.info("Starting training process...")
        
        pad_id = self.tokenizer.hf_tokenizers.pad_token_id
        mask_id =  self.tokenizer.hf_tokenizers.mask_token_id

        update_size = 1 if (len(self) // 10 == 0) else len(self) // 10

        train_len = len(self)

        print("Update size: ", update_size, " Train set size:", train_len)

        for epoch in range(start_epoch, num_epoch):
            start_time = datetime.now().time()
            self.model.train()
            total_loss = 0.0
            losses_per_batch = []
            total_acc = 0.0
            lm_accuracy_per_batch = []

            print ("epoch #: ", epoch)
            for i, data in enumerate(self, 0):
                tokens, masked_for_pred, e1_e2_start, _, blank_labels, _, _, _, _, _ = data
                masked_for_pred = masked_for_pred[(masked_for_pred != pad_id)]
                
                print("processing record : ", i)
                if masked_for_pred.shape[0] == 0:
                    print('Empty dataset, skipping...')
                    continue

                attention_mask = (tokens != pad_id).float()
                token_type_ids = torch.zeros((tokens.shape[0], tokens.shape[1])).long()
                
                if self.is_cuda_available:
                    tokens = tokens.to(device)
                    masked_for_pred = masked_for_pred.to(device)
                    attention_mask = attention_mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                
                blanks_logits, lm_logits = self.model(tokens, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None, \
                          e1_e2_start=e1_e2_start, pooled_output=None)

                token_mask_matches = (tokens == mask_id) 
               
                if (i % update_size) == (update_size - 1):
                    verbose = True
                else:
                    verbose = False

                #blank_labels = torch.zeros((blanks_logits.size()))

                input_matched_lm_logits = lm_logits[0][token_mask_matches]

                loss = criterion(input_matched_lm_logits, blanks_logits, masked_for_pred, blank_labels, verbose=verbose)
                loss = loss/gradient_acc_steps
              
                loss.backward()
                
                if (i % gradient_acc_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
          
                total_acc += Two_Headed_Loss.evaluate_results(input_matched_lm_logits, blanks_logits, 
                                    masked_for_pred, blank_labels, \
                                    self.tokenizer.hf_tokenizers, print_=verbose)
                
                if (i % update_size) == (update_size - 1):
                    losses_per_batch.append(gradient_acc_steps * total_loss/ update_size)
                    lm_accuracy_per_batch.append(total_acc/update_size)
                    print("Losses per batch: " , str(losses_per_batch))
                    print("LM acc per batch: " , str(lm_accuracy_per_batch))
                    print('[Epoch: %d, %5d/ %d points] total loss, lm accuracy per batch: %.3f, %.3f' %
                        (epoch + 1, (i + 1), train_len, losses_per_batch[-1], lm_accuracy_per_batch[-1]))
                    total_loss = 0.0; total_acc = 0.0
                    logging.info("Last batch samples (pos, neg): %d, %d" % ((blank_labels.squeeze() == 1).sum().item(),\
                                                                        (blank_labels.squeeze() == 0).sum().item()))
               
            scheduler.step()
            losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
            accuracy_per_epoch.append(sum(lm_accuracy_per_batch)/len(lm_accuracy_per_batch))
         

            end_time = datetime.now().time()
            
            print("Epoch finished, took " + str(datetime.combine(date.today(), end_time) - datetime.combine(date.today(), start_time) ) + " seconds")
            print("Losses at Epoch %d: %.7f" % (epoch + 1, losses_per_epoch[-1]))
            print("Accuracy at Epoch %d: %.7f" % (epoch + 1, accuracy_per_epoch[-1]))

            if accuracy_per_epoch[-1] > best_pred:
                best_pred = accuracy_per_epoch[-1]
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': self.model.state_dict(),\
                        'best_acc': accuracy_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict()
                    }, os.path.join("./data/" , "test_model_best_%s.pth.tar" % "BERT_uncased"))
        
            if (epoch % 1) == 0:
                save_bin_file("test_losses_per_epoch_%s.pkl" % "BERT_uncased", losses_per_epoch)
                save_bin_file("test_accuracy_per_epoch_%s.pkl" % "BERT_uncased", accuracy_per_epoch)
                torch.save({
                        'epoch': epoch + 1,\
                        'state_dict': self.model.state_dict(),\
                        'best_acc': accuracy_per_epoch[-1],\
                        'optimizer' : optimizer.state_dict(),\
                        'scheduler' : scheduler.state_dict()
                    }, os.path.join("./data/" , "test_checkpoint_%s.pth.tar" % "BERT_uncased"))
        
    def pretrain_dataset(self):
       self.ENT1_list = list(self.rel_data.relations_dataframe["ent1"].unique())
       self.ENT2_list = list(self.rel_data.relations_dataframe["ent2"].unique())

       self.tokenizer.hf_tokenizers.add_tokens(["[BLANK]", "[ENT1]", "[ENT2]", "[/ENT1]", "[/ENT2]"])
       
       self.tokenizer.hf_tokenizers.convert_tokens_to_ids("[ENT1]")
       self.tokenizer.hf_tokenizers.convert_tokens_to_ids("[ENT2]")

       self.cls_token = self.tokenizer.hf_tokenizers.cls_token
       self.sep_token = self.tokenizer.hf_tokenizers.sep_token

       self.ENT1_token_id = self.tokenizer.hf_tokenizers.encode("[ENT1]")[1:-1][0]
       self.ENT2_token_id = self.tokenizer.hf_tokenizers.encode("[ENT2]")[1:-1][0]
       self.ENT1_s_token_id = self.tokenizer.hf_tokenizers.encode("[/ENT1]")[1:-1][0]
       self.ENT2_s_token_id = self.tokenizer.hf_tokenizers.encode("[/ENT2]")[1:-1][0]

       self.padding_seq = Pad_Sequence(seq_pad_value=self.tokenizer.hf_tokenizers.pad_token_id,\
                              label_pad_value=self.tokenizer.hf_tokenizers.pad_token_id,\
                              label2_pad_value=-1)
       
       save_bin_file(file_name="BERT_tokenizer_relation_extraction.dat", data=self.tokenizer)

    def put_blanks(self, relations_dataset):
        
        blank_ent1 = numpy.random.uniform()
        blank_ent2 = numpy.random.uniform()
        
        sentence_token_span, ent1, ent2 = relations_dataset
        
        if blank_ent1 >= self.alpha:
            relations_dataset = (sentence_token_span, "[BLANK]", ent2)
        
        if blank_ent2 >= self.alpha:
            relations_dataset = (sentence_token_span, ent1, "[BLANK]")
            
        return relations_dataset

    def tokenize(self, relations_dataset: Series):

        (tokens, span_1_pos, span_2_pos), ent1_text, ent2_text = relations_dataset 

        tokens = [token.lower() for token in tokens if tokens != '[BLANK]']

        forbidden_indices = [i for i in range(span_1_pos[0], span_1_pos[1])] + [i for i in range(span_2_pos[0], span_2_pos[1])]

        pool_indices = [ i for i in range(len(tokens)) if i not in forbidden_indices ]

        masked_indices = numpy.random.choice(pool_indices, \
                                          size=round(self.mask_probability * len(pool_indices)), \
                                          replace=False)
   
        masked_for_pred = [token.lower() for idx, token in enumerate(tokens) if (idx in masked_indices)]

        tokens = [token if (idx not in masked_indices) else self.tokenizer.hf_tokenizers.mask_token for idx, token in enumerate(tokens)]

        if (ent1_text == "[BLANK]") and (ent2_text != "[BLANK]"):
            tokens = [self.cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]" ,"[BLANK]", "[/ENT1]"] + \
                tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]"] + tokens[span_2_pos[0]:span_2_pos[1]] + ["[/ENT2]"] + tokens[span_2_pos[1]:] + [self.sep_token]
        
        elif (ent1_text == "[BLANK]") and (ent2_text == "[BLANK]"):
            tokens = [self.cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]" ,"[BLANK]", "[/ENT1]"] + \
                tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]", "[BLANK]", "[/ENT2]"] + tokens[span_2_pos[1]:] + [self.sep_token]
        
        elif (ent1_text != "[BLANK]") and (ent2_text == "[BLANK]"):
            tokens = [self.cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]"] + tokens[span_1_pos[0]:span_1_pos[1]] + ["[/ENT1]"] + \
                tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]", "[BLANK]", "[/ENT2]"] + tokens[span_2_pos[1]:] + [self.sep_token]
        
        elif (ent1_text != "[BLANK]") and (ent2_text != "[BLANK]"):
            tokens = [self.cls_token] + tokens[:span_1_pos[0]] + ["[ENT1]"] + tokens[span_1_pos[0]:span_1_pos[1]] + ["[/ENT1]"] + \
                tokens[span_1_pos[1]:span_2_pos[0]] + ["[ENT2]"] + tokens[span_2_pos[0]:span_2_pos[1]] + ["[/ENT2]"] + tokens[span_2_pos[1]:] + [self.sep_token]

        ent1_ent2_start = ([i for i, e in enumerate(tokens) if e == "[ENT1]"] [0] , [i for i, e in enumerate(tokens) if e == "[ENT2]"] [0])

        token_ids = self.tokenizer.hf_tokenizers.convert_tokens_to_ids(tokens)
        masked_for_pred = self.tokenizer.hf_tokenizers.convert_tokens_to_ids(masked_for_pred)

        # print(tokens)

        return token_ids, masked_for_pred, ent1_ent2_start

    def __len__(self):
        return len(self.rel_data.relations_dataframe)

    def __getitem__(self, index):
        relation, ent1_text, ent2_text = self.rel_data.relations_dataframe.iloc[index]

        logging.debug("relation : ", str(relation), " | ent1_text: " + ent1_text, "| ent2_text" + ent2_text)
        logging.debug("\n")

        pool = self.rel_data.relations_dataframe[((self.rel_data.relations_dataframe["ent1"] == ent1_text) & (self.rel_data.relations_dataframe["ent2"] == ent2_text))].index
        
        pool.append(self.rel_data.relations_dataframe[((self.rel_data.relations_dataframe["ent1"] == ent2_text) & (self.rel_data.relations_dataframe["ent2"] == ent1_text))].index)
        
        pos_idxs = numpy.random.choice(pool, size=min(int(self.batch_size//2), len(pool)), replace=False)

        neg_idxs = []

        if numpy.random.uniform() > 0.5:   
            pool = self.rel_data.relations_dataframe[((self.rel_data.relations_dataframe["ent1"] != ent1_text) | \
                                                     (self.rel_data.relations_dataframe["ent2"] != ent2_text))].index
        else:
            if numpy.random.uniform() > 0.5: # share e1 but not e2
                pool = self.rel_data.relations_dataframe[(( self.rel_data.relations_dataframe['ent1'] == ent1_text) & \
                     (self.rel_data.relations_dataframe['ent2'] != ent2_text))].index
            else: # share e2 but not e1
                pool = self.rel_data.relations_dataframe[(( self.rel_data.relations_dataframe['ent1'] != ent1_text) & \
                     (self.rel_data.relations_dataframe['ent2'] == ent2_text))].index

            if len(pool) == 0:
                    pool = self.rel_data.relations_dataframe[((self.rel_data.relations_dataframe["ent1"] != ent1_text) | \
                        (self.rel_data.relations_dataframe["ent2"] != ent2_text))].index
                                                     
        neg_idxs = numpy.random.choice(pool, size=min(int(self.batch_size//2), len(pool)), replace=False)
        Q = 1/len(pool)

        logging.debug(" Pos Idx: " + str(pos_idxs))
        logging.debug(" Neg Idx: " + str(neg_idxs))

        batch = []
        ## process positive sample
        pos_df = self.rel_data.relations_dataframe.loc[pos_idxs]
        for idx, row in pos_df.iterrows():
            relation, ent1_text, ent2_text = row[0], row[1], row[2]
            relation_tokens, masked_for_pred, e1_e2_start = self.tokenize(self.put_blanks((relation, ent1_text, ent2_text)))
            relation_tokens = torch.LongTensor(relation_tokens)

            masked_for_pred = torch.LongTensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
            batch.append((relation_tokens, masked_for_pred, e1_e2_start, torch.FloatTensor([1.0]),\
                            torch.LongTensor([1])))
        
        ## process negative samples
        negs_df = self.rel_data.relations_dataframe.loc[neg_idxs]
        for idx, row in negs_df.iterrows():
            relation, ent1_text, ent2_text = row[0], row[1], row[2]
            relation_tokens, masked_for_pred, e1_e2_start = self.tokenize(self.put_blanks((relation, ent1_text, ent2_text)))
            relation_tokens = torch.LongTensor(relation_tokens)
            masked_for_pred = torch.LongTensor(masked_for_pred)
            e1_e2_start = torch.tensor(e1_e2_start)
    
            batch.append((relation_tokens, masked_for_pred, e1_e2_start, torch.FloatTensor([Q]), torch.LongTensor([0])))

        batch = self.padding_seq(batch)

        return batch


class Two_Headed_Loss(nn.Module):
    '''
    Implements LM Loss and matching-the-blanks loss concurrently
    '''
    def __init__(self, lm_ignore_idx, use_logits=False, normalize=False):
        super(Two_Headed_Loss, self).__init__()
        self.lm_ignore_idx = lm_ignore_idx
        self.LM_criterion = nn.CrossEntropyLoss(ignore_index=lm_ignore_idx)
        self.use_logits = use_logits
        self.normalize = normalize
        
        if not self.use_logits:
            self.BCE_criterion = nn.BCELoss(reduction='mean')
        else:
            self.BCE_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    def p_(self, f1_vec, f2_vec):
        if self.normalize:
            factor = 1 / (torch.norm(f1_vec)*torch.norm(f2_vec))
        else:
            factor = 1.0
        
        if not self.use_logits:
            p = 1/(1 + torch.exp(-factor*torch.dot(f1_vec, f2_vec)))
        else:
            p = factor*torch.dot(f1_vec, f2_vec)
        return p
    
    def dot_(self, f1_vec, f2_vec):
        return -torch.dot(f1_vec, f2_vec)
    
    def forward(self, lm_logits, blank_logits, lm_labels, blank_labels, verbose=False):
        '''
        lm_logits: (batch_size, sequence_length, hidden_size)
        lm_labels: (batch_size, sequence_length, label_idxs)
        blank_logits: (batch_size, enumerate)
        blank_labels: (batch_size, 0 or 1)
        '''
        pos_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 1]
        neg_idxs = [i for i, l in enumerate(blank_labels.squeeze().tolist()) if l == 0]
        
        if len(pos_idxs) > 1:
            # positives
            pos_logits = []
            for pos1, pos2 in combinations(pos_idxs, 2):
                pos_logits.append(self.p_(blank_logits[pos1, :], blank_logits[pos2, :]))
            pos_logits = torch.stack(pos_logits, dim=0)
            pos_labels = [1.0 for _ in range(pos_logits.shape[0])]
        else:
            pos_logits, pos_labels = torch.FloatTensor([]), []
        
        # negatives
        neg_logits = []
        for pos_idx in pos_idxs:
            for neg_idx in neg_idxs:
                neg_logits.append(self.p_(blank_logits[pos_idx, :], blank_logits[neg_idx, :]))

        neg_logits = torch.stack(neg_logits, dim=0)

        neg_labels = [0.0 for _ in range(neg_logits.shape[0])]
        blank_labels_ = torch.FloatTensor(pos_labels + neg_labels)
        
        
        if blank_logits.is_cuda:
            blank_labels_ = blank_labels_.cuda()
            pos_logits = pos_logits.cuda()
            neg_logits = neg_logits.cuda()

        lm_loss = self.LM_criterion(lm_logits, target=lm_labels.long())


        blank_loss = self.BCE_criterion(torch.cat([pos_logits, neg_logits], dim=0), \
                                        blank_labels_)

        if verbose:
            print("LM loss, blank_loss for last batch: %.5f, %.5f" % (lm_loss, blank_loss))
           
        total_loss = lm_loss + blank_loss
        return total_loss

    @classmethod
    def load_state(net, optimizer, scheduler, model_name="BERT", load_best=False):
        """ Loads saved model and optimizer states if exists """
        base_path = "./data/"
        amp_checkpoint = None
        checkpoint_path = os.path.join(base_path,"test_checkpoint_%s.pth.tar" % model_name)
        best_path = os.path.join(base_path,"test_model_best_%s.pth.tar" % model_name)
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
            amp_checkpoint = checkpoint['amp']
            logging.info("Loaded model and optimizer.")    
        return start_epoch, best_pred, amp_checkpoint

    @classmethod
    def load_results(cls, model_name="BERT"):
        """ Loads saved results if exists """
        losses_path = "./data/test_losses_per_epoch_%s.pkl" % model_name
        accuracy_path = "./data/test_accuracy_per_epoch_%s.pkl" % model_name
        if os.path.isfile(losses_path) and os.path.isfile(accuracy_path):
            losses_per_epoch = load_bin_file("test_losses_per_epoch_%s.pkl" % model_name)
            accuracy_per_epoch = load_bin_file("test_accuracy_per_epoch_%s.pkl" % model_name)
            logging.info("Loaded results buffer")
        else:
            losses_per_epoch, accuracy_per_epoch = [], []
        return losses_per_epoch, accuracy_per_epoch
    
    @classmethod
    def evaluate_results(cls, lm_logits, blanks_logits, masked_for_pred, blank_labels, tokenizer, print_=False):
        '''
        evaluate must be called after loss.backward()
        '''
     
        lm_logits_pred_ids = torch.softmax(input=lm_logits, dim=-1).max(1)[1]
        lm_accuracy = ((lm_logits_pred_ids == masked_for_pred).sum().float()/len(masked_for_pred)).item()
        
        if print_:
            print("Predicted masked tokens: \n")
            print(tokenizer.decode(lm_logits_pred_ids.cpu().numpy() if lm_logits_pred_ids.is_cuda else \
                                lm_logits_pred_ids.numpy()))
            print("\n Masked labels tokens: \n")
            print(tokenizer.decode(masked_for_pred.cpu().numpy() if masked_for_pred.is_cuda else \
                                masked_for_pred.numpy()))
  
        return lm_accuracy

class Pad_Sequence():
    """
        collate_fn for dataloader to collate sequences of different lengths into a fixed length batch
        Returns padded x sequence, y sequence, x lengths and y lengths of batch
    """
    def __init__(self, seq_pad_value, label_pad_value=1, label2_pad_value=-1, label3_pad_value=-1, label4_pad_value=-1):
        self.seq_pad_value = seq_pad_value
        self.label_pad_value = label_pad_value
        self.label2_pad_value = label2_pad_value
        self.label3_pad_value = label3_pad_value
        self.label4_pad_value = label4_pad_value
        
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        seqs = [x[0] for x in sorted_batch]
        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=self.seq_pad_value)
        x_lengths = torch.LongTensor([len(x) for x in seqs])
        
        labels = list(map(lambda x: x[1], sorted_batch))
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_value)
        y_lengths = torch.LongTensor([len(x) for x in labels])
        
        labels2 = list(map(lambda x: x[2], sorted_batch))
        labels2_padded = pad_sequence(labels2, batch_first=True, padding_value=self.label2_pad_value)
        y2_lengths = torch.LongTensor([len(x) for x in labels2])
        
        labels3 = list(map(lambda x: x[3], sorted_batch))
        labels3_padded = pad_sequence(labels3, batch_first=True, padding_value=self.label3_pad_value)
        y3_lengths = torch.LongTensor([len(x) for x in labels3])
        
        labels4 = list(map(lambda x: x[4], sorted_batch))
        labels4_padded = pad_sequence(labels4, batch_first=True, padding_value=self.label4_pad_value)
        y4_lengths = torch.LongTensor([len(x) for x in labels4])

        return seqs_padded, labels_padded, labels2_padded, labels3_padded, labels4_padded, \
               x_lengths, y_lengths, y2_lengths, y3_lengths, y4_lengths