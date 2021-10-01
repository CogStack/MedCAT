from itertools import permutations
import logging
import os
import numpy
import spacy
import datasets
import logging
import pandas
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

from tqdm.autonotebook import tqdm  

Doc.set_extension("relations", default=[], force=True)
Doc.set_extension("ents", default=[], force=False)

class RelationalModel(object):

    def __init__(self, docs, spacy_nlp):
        self.docs = docs
        self.predictions = []
        self.spacy_nlp = spacy_nlp

        self.create_base_relations()

    def get_model(self):
        return self.docs, self.predictions
    
    def create_base_relations(self):
        for doc_id, doc in self.docs.items():
            if len(doc._.relations) == 0:
                doc._.ents = doc.ents
                #doc._.relations = self.get_instances(doc)
                doc._.relations = self.create_base_relation(doc.text, self.spacy_nlp)
    
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
                       relation_instances.append((self.relation_labels[0], ent1, ent2))
                       
       return relation_instances

    def get_all_instances(self):
        relation_instances = []
        for doc_id, doc in self.docs.items():
            relation_instances.extend(doc._.relations)
        return relation_instances
    
    def get_subject_objects(self, entity):
        root = entity.sent.root
        subject = None; objs = []; pairs = []
        for child in list(root.children):
            if child.dep_ in ["nsubj", "nsubjpass", "nmod:poss"]:
                subject = child; 
            elif child.dep_ in ["compound", "dobj", "conj", "ccomp"]: # ["dobj", "attr", "prep", "ccomp", "compound"]:
                objs.append(child)

        if (subject is not None) and (len(objs) > 0):
            for a, b in permutations([subject] + [obj for obj in objs], 2):
                a_ = [w for w in a.subtree]
                b_ = [w for w in b.subtree]
                pairs.append((a_[0] if (len(a_) == 1) else a_ , b_[0] if (len(b_) == 1) else b_))
                
        return pairs
        
    def create_base_relation(self, raw_text, nlp, window_size=100):
        '''
        Input: Chunk of raw text
        Output: modified corpus of triplets (relation statement, entity1, entity2)
        '''
        logging.info("Processing sentences...")
        sents_doc = nlp(raw_text)
        ents = sents_doc.ents # get entities
        
        logging.info("Processing relation statements by entities...")
        entities_of_interest = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", \
                                "LAW", "LANGUAGE", "ENTITY"]
        length_doc = len(sents_doc)
        relations = []; ents_list = []
        for i in tqdm(range(len(ents))):
            e1 = ents[i]
            e1start = e1.start; e1end = e1.end
            if e1.label_ not in entities_of_interest:
                continue
            if re.search("[\d+]", e1.text): # entities should not contain numbers
                continue
            
            for j in range(1, len(ents) - i):
                e2 = ents[i + j]
                e2start = e2.start; e2end = e2.end
                if e2.label_ not in entities_of_interest:
                    continue
                if re.search("[\d+]", e2.text): # entities should not contain numbers
                    continue
                if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
                    continue
                
                if (1 <= (e2start - e1end) <= window_size): # check if next nearest entity within window_size
                    # Find start of sentence
                    punc_token = False
                    start = e1start - 1
                    if start > 0:
                        while not punc_token:
                            punc_token = sents_doc[start].is_punct
                            start -= 1
                            if start < 0:
                                break
                        left_r = start + 2 if start > 0 else 0
                    else:
                        left_r = 0
                    
                    # Find end of sentence
                    punc_token = False
                    start = e2end
                    if start < length_doc:
                        while not punc_token:
                            punc_token = sents_doc[start].is_punct
                            start += 1
                            if start == length_doc:
                                break
                        right_r = start if start < length_doc else length_doc
                    else:
                        right_r = length_doc
                    
                    if (right_r - left_r) > window_size: # sentence should not be longer than window_size
                        continue
                    
                    x = [token.text for token in sents_doc[left_r:right_r]]
                    
                    ### empty strings check ###
                    for token in x:
                        assert len(token) > 0
                    assert len(e1.text) > 0
                    assert len(e2.text) > 0
                    assert e1start != e1end
                    assert e2start != e2end
                    assert (e2start - e1end) > 0
                    
                    r = (x, (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r))
                    relations.append((r, e1.text, e2.text))
                    ents_list.append((e1.text, e2.text))
        
        logging.info("Processing relation statements by dependency tree parsing...")
        doc_sents = [s for s in sents_doc.sents]
        for sent_ in tqdm(doc_sents, total=len(doc_sents)):
            if len(sent_) > (window_size + 1):
                continue
            
            left_r = sent_[0].i
            pairs = self.get_subject_objects(sent_)
            
            if len(pairs) > 0:
                for pair in pairs:
                    e1, e2 = pair[0], pair[1]
                    
                    if (len(e1) > 2) or (len(e2) > 2): # don't want entities that are too long
                        continue
                    
                    e1text, e2text = " ".join(w.text for w in e1) if isinstance(e1, list) else e1.text,\
                                        " ".join(w.text for w in e2) if isinstance(e2, list) else e2.text
                    e1start, e1end = e1[0].i if isinstance(e1, list) else e1.i, e1[-1].i + 1 if isinstance(e1, list) else e1.i + 1
                    e2start, e2end = e2[0].i if isinstance(e2, list) else e2.i, e2[-1].i + 1 if isinstance(e2, list) else e2.i + 1
                    if (e1end < e2start) and ((e1text, e2text) not in ents_list):
                        assert e1start != e1end
                        assert e2start != e2end
                        assert (e2start - e1end) > 0
                        r = ([w.text for w in sent_], (e1start - left_r, e1end - left_r), (e2start - left_r, e2end - left_r))
                        relations.append((r, e1text, e2text))
                        ents_list.append((e1text, e2text))
        
        return relations

class RelationExtraction(object):

    name : str = "rel"

    def __init__(self, docs, vocab: Vocab, config: Config = None, rel_model: RelationalModel = None, spacy_model = None, tokenizer = None, embeddings=None, model="ltsm", threshold: float = 0.1):
       self.vocab = vocab
       self.config = config
       self.rel_model = rel_model
       self.model = model
       self.cfg = {"labels": [], "threshold": threshold }
       self.tokenizer = tokenizer
       self.embeddings = embeddings

       if self.tokenizer is None:
           self.tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(pretrained_model_name_or_path="emilyalsentzer/Bio_ClinicalBERT"))

       if self.embeddings is None:
           embeddings = numpy.load(os.path.join("./", "embeddings.npy"), allow_pickle=False)
           self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

       self.spacy_nlp = spacy.load("en_core_sci_lg") if spacy_model is None else spacy.load(spacy_model)

       if rel_model is None:
           self.rel_model = RelationalModel(docs, self.spacy_nlp)

       self.device = torch.device("cpu")
    
    def pretrain_dataset(self):
       self.df = pandas.DataFrame(self.rel_model.get_all_instances(), columns=['r','e1','e2'])
       self.e1s = list(self.df['e1'].unique())
       self.e2s = list(self.df['e2'].unique())

       self.tokenizer.hf_tokenizers.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])
       e1_id = self.tokenizer.convert_tokens_to_ids('[E1]')
       e2_id = self.tokenizer.convert_tokens_to_ids('[E2]')
       

       # train_set = pretrain_dataset(args, D, batch_size=args.batch_size)
       # train_length = len(train_set)






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