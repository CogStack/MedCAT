import logging
import os
import numpy
from numpy.random import permutation
import spacy
import datasets
import os
import torch
import torch.nn
import regex as re
from torch.nn.modules.module import T

from medcat.preprocessing.tokenizers import TokenizerWrapperBERT, TokenizerWrapperBPE
from medcat.pipe import Pipe
from medcat.vocab import Vocab
from medcat.config import Config
from medcat.utils.models import LSTM
from spacy.tokens import Doc
from typing import Dict, Iterable, List, Set, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments


Doc.set_extension("relations", default=[], force=True)

class RelationalModel(object):

    def __init__(self, docs):
        self.docs = docs
        self.predictions = []
        self.default_blank_relation = "[BLANK]"

        self.create_base_relations()

    def get_model(self):
        return self.docs, self.predictions
    
    def create_base_relations(self):
        for doc_id, doc in self.docs.items():
            if len(doc._.relations) == 0:
                doc._.relations = self.get_instances(doc)
    
    def __call__(self, doc_id):
        if doc_id in self.docs.keys():
            return self.docs[doc_id]._.relations
        return []

    def get_instances(self, doc, window_size=250) -> List[Tuple]:
       """  
            doc : SpacyDoc

            window_size : int
                Character distance between any two entities start positions.

            Creates a list of tuples based on pairs of entities detected (relation, ent1, ent2) for one spacy document.
       """
       relation_instances = []
       for ent1 in doc.ents:
           for ent2 in doc.ents:
               if ent1 != ent2:
                   if window_size and abs(ent2.start - ent1.start) <= window_size:
                       relation_instances.append((self.default_blank_relation, ent1, ent2))

       return relation_instances
    
    def get_all_instances(self):
        relation_instances = []
        for doc in self.docs:
            relation_instances.extend(doc._.relations)
        return relation_instances
    
    def get_subject_objects(self, entity):
        root = entity.root
        subject = None; objs = []; pairs = []
        for child in root.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                if len(re.findall("[a-z]+",child.text.lower())) > 0: # filter out all numbers/symbols
                    subject = child; 
            elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
                objs.append(child); 
        if (subject is not None) and (len(objs) > 0):
            for a, b in permutation([subject] + [obj for obj in objs], 2):
                a_ = [w for w in a.subtree]
                b_ = [w for w in b.subtree]
                pairs.append((a_[0] if (len(a_) == 1) else a_ , b_[0] if (len(b_) == 1) else b_))
                
        return pairs


class RelationExtraction(object):

    name : str = "rel"

    def __init__(self, vocab: Vocab, config: Config = None, model: RelationalModel = None,  tokenizer = None, embeddings=None, threshold: float = 0.1):
       self.vocab = vocab
       self.config = config
       self.model = model
       self.cfg = {"labels": [], "threshold": threshold }
       self.tokenizer = tokenizer
       self.embeddings = embeddings

       if self.tokenizer is None:
           self.tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased"))

       if self.embeddings is None:
           embeddings = numpy.load(os.path.join("./", "embeddings.npy"), allow_pickle=False)
           self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

       self.spacy_nlp = spacy.load("en_core_sci_lg")


    def __call__(self, doc_id) -> Doc:
        
        total_instances = len(self.model.get_instances(doc_id))
        doc = self.model.docs[doc_id]

        if total_instances == 0:
            logging.info("Could not determine any instances in doc - returning doc as is.")
            return self.model.docs[doc_id]


        return doc

    def predict(self, docs: Dict[int,Doc]) -> float:
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        all_instances = self.model.get_all_instances()

        total_instances = len(all_instances)
        
        if total_instances == 0:
            logging.info("Could not determine any instances in any docs - can not make any predictions.")
        
        scores = self.model.predict(docs)

        return self.model.ops.asarray(scores)