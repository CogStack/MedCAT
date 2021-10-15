
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
import torch.optim
from torch.nn.modules.module import T
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertConfig
from transformers.configuration_utils import PretrainedConfig
from itertools import permutations
from pandas.core.series import Series

from medcat.preprocessing.tokenizers import TokenizerWrapperBERT
from medcat.vocab import Vocab
from medcat.config import Config
from spacy.tokens import Doc
from typing import Dict, Iterable, List, Set, Tuple
from transformers import AutoTokenizer

from tqdm.autonotebook import tqdm  

Doc.set_extension("relations", default=[], force=True)
Doc.set_extension("ents", default=[], force=False)

class RelData(object):

    def __init__(self, docs):
        self.docs = docs
        self.predictions = []
        self.create_base_relations()
        self.relations_dataframe = pandas.DataFrame()

    def get_model(self):
        return self.docs, self.predictions
    
    def create_base_relations(self):
        for doc_id, doc in self.docs.items():
            if len(doc._.relations) == 0:
                doc._.ents = doc.ents
                doc._.relations = self.get_instances(doc)
    
    def __call__(self, doc_id):
        if doc_id in self.docs.keys():
            return self.docs[doc_id]._.relations
        return []

    def get_instances(self, doc, window_size=250):
        """  
            doc : SpacyDoc
            window_size : int, Character distance between any two entities start positions.
            Creates a list of tuples based on pairs of entities detected (relation, ent1, ent2) for one spacy document.
        """
        relation_instances = []
        
        doc_length = len(doc)

        j = 1
        for ent1 in doc.ents:
            j += 1
            for ent2 in doc.ents[j:]:
                if ent1 != ent2 and window_size and abs(ent2.start - ent1.start) <= window_size:
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
                        pass
                    
                    sentence_window_tokens = [token.text for token in doc[left_sentence_context:right_sentence_context]]

                    sentence_token_span = (sentence_window_tokens,
                                     (ent1.start - left_sentence_context, ent1.end - left_sentence_context ),
                                     (ent2.start - left_sentence_context, ent2.end - left_sentence_context)
                       )

                    relation_instances.append((sentence_token_span, ent1.text, ent2.text))

        return relation_instances

    def get_all_instances(self):
        relation_instances = []
        for doc_id, doc in self.docs.items():
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

    def __init__(self, docs, vocab: Vocab, config: PretrainedConfig = None, rel_data: RelData = None, spacy_model : str = None, tokenizer = None, embeddings=None, model=None, threshold: float = 0.1):
    
       self.vocab = vocab
       self.config = config
       self.rel_data = rel_data
       self.cfg = {"labels": [], "threshold": threshold }
       self.tokenizer = tokenizer
       self.embeddings = embeddings
       self.learning_rate = 0.1

       if self.config is None:
           self.config = BertConfig.from_pretrained("bert-base-uncased")  

       if model is None:
           self.model = BertModel.from_pretrained("bert-base-uncased", config=config, add_pooling_layer=False)  

       if self.tokenizer is None:
            if os.path.isfile("./BERT_tokenizer_relation_extraction.dat"):
                self.load_tokenizer()
            else:
                self.tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path="emilyalsentzer/Bio_ClinicalBERT"))

       if self.embeddings is None:
           embeddings = numpy.load(os.path.join("./", "embeddings.npy"), allow_pickle=False)
           self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

       self.spacy_nlp = spacy.load("en_core_sci_lg") if spacy_model is None else spacy.load(spacy_model)

       if rel_data is None:
           self.rel_data = RelData(docs)

       self.alpha = 0.7
       self.mask_probability = 0.15

    def train(self):
        self.model.resize_token_embeddings(len(self.tokenizer.hf_tokenizers))

        optimizer = torch.optim.Adam([{"params":self.model.parameters(), "lr": self.learning_rate}])
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,6,8,12,15,18,20,22,24,26,30], gamma=0.8)
        start_epoch, best_pred, amp_checkpoint = self.load_state(self.model, optimizer, scheduler, load_best=False)  

    def load_bin_file(filename):
        with open(os.path.join("./", filename), 'rb') as f:
            data = pickle.load(f)
        return data

    def save_tokenizer(self, file_name, tokenizer):
        with open(os.path.join("", file_name), "wb") as f:
            pickle.dump(tokenizer, f)
    
    def load_tokenizer(self, file_name="./BERT_tokenizer_relation_extraction.dat"):
        with open(os.path.join("", file_name), "rb") as f:
            self.tokenizer = pickle.load(f)

    def pretrain_dataset(self):
       self.rel_data.relations_dataframe = pandas.DataFrame(self.rel_data.get_all_instances(), columns=["relation", "ent1", "ent2"])

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
       
       self.save_tokenizer("BERT_tokenizer_relation_extraction.dat", self.tokenizer)

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

        tokens =  [token.lower() for token in tokens if tokens != '[BLANK]']

        forbidden_indices = [i for i in range(span_1_pos[0], span_1_pos[1])] + [i for i in range(span_2_pos[0], span_2_pos[1])]

        pool_indices = [ i for i in range(len(tokens)) if i not in forbidden_indices ]

        masked_indices = numpy.random.choice(pool_indices, \
                                          size=round(self.mask_probability * len(pool_indices)), \
                                          replace=False)
   
        masked_for_pred = [token for idx, token in enumerate(tokens) if (idx in masked_indices)]

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

        tagged_tokens = self.tokenizer.hf_tokenizers.convert_tokens_to_ids(tokens)
        masked_for_pred = self.tokenizer.hf_tokenizers.convert_tokens_to_ids(masked_for_pred)
        

        return tagged_tokens, masked_for_pred, ent1_ent2_start

    def __getitem__(self, index):
        relation, ent1_text, ent2_text = self.rel_data.relations_dataframe.iloc[index]
        tokens, masked_for_pred, ent1_ent2_start = self.tokenize(self.put_blanks((relation, ent1_text, ent2_text)))
        tokens = torch.tensor(tokens)
        masked_for_pred = torch.tensor(masked_for_pred)
        ent1_ent2_start = torch.tensor(ent1_ent2_start)
      
        return tokens, masked_for_pred, ent1_ent2_start, ent1_text, ent2_text

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