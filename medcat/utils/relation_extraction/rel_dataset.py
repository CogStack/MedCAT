
from ast import literal_eval
from typing import Iterable, List, Tuple, Dict
import torch
from torch.utils.data import Dataset
from spacy.tokens import Doc
import pandas

from medcat.cdb import CDB
from medcat.config_re import ConfigRE
from medcat.utils.relation_extraction.tokenizer import TokenizerWrapperBERT

class RelData(Dataset):

    Doc.set_extension("relations", default=[], force=True)

    def __init__(self, tokenizer : TokenizerWrapperBERT, config : ConfigRE, cdb : CDB):
        self.cdb = cdb
        self.config = config
        self.tokenizer = tokenizer
        self.blank_label_id = -1
        self.dataset = []
        self.window_size = self.config.general["window_size"]
        self.ent_context_left = self.config.general["ent_context_left"]
        self.ent_context_right = self.config.general["ent_context_right"]

    def generate_base_relations(self, docs : Iterable[Doc]) -> Iterable[List[Tuple]]:
        '''
            Generate relations from Spacy CAT docs,
        '''
        output_relations = []
        for doc_id, doc in docs.items():
            output_relations.append(self.create_base_relations_from_doc(doc,
                                                          doc_id=doc_id,
                                                          ent_context_left=self.config.general['entity_context_left'],
                                                          ent_context_right=self.config.general['entity_context_right']
                                                         ))
        
        return output_relations

    def create_base_relations_from_csv(self, csv_path):
        # assumes the columns are as follows ["relation_token_span_ids", "ent1_ent2_start", "ent1", "ent2", "label", "label_id", "ent1_type", "ent2_type", "ent1_id", "ent2_id", "ent1_cui", "ent2_cui", "doc_id", "sents"],
        #  last column is the actual source text

        df = pandas.read_csv(csv_path, index_col=False, sep='\t', encoding='utf-8')
        df["ent1_ent2_start"] = df["ent1_ent2_start"].apply(lambda x: literal_eval(str(x)))
        df["relation_token_span_ids"] = df["relation_token_span_ids"].apply(lambda x: literal_eval(str(x)))

        df = df.drop('sents', 1)

        n_classes, unique_labels, labels2idx, idx2label = self.get_labels(df["label"])

        output_relations = df.values.tolist() # df.to_records(index=False).tolist()
        
        # replace label_id with actual detected label number
        for idx in range(len(output_relations)):
            output_relations[idx][5] = labels2idx[output_relations[idx][4]]

        #print(output_relations[0])
        return { "output_relations": output_relations, "n_classes": n_classes, "unique_labels": unique_labels, "labels2idx": labels2idx, "idx2label": idx2label}
       

    def create_base_relations_from_doc(self, doc, doc_id) -> Dict:
        """  
            doc : SpacyDoc
            window_size : int, Character distance between any two entities start positions.
            Creates a list of tuples based on pairs of entities detected (relation, ent1, ent2) for one spacy document.

            Returns:
                relations (List[list]) :
                   data columns: ["relation_token_span", "ent1", "ent2", "label", "label_id", "ent1_type", "ent2_type", "ent1_id", "ent2_id", "ent1_cui", "ent2_cui", "doc_id"]
        """
        relation_instances = []
        doc_length = len(doc)

        chars_to_exclude = ":!@#$%^&*()-+?_=,<>/"

        j = 1
        for ent1 in doc.ents:
            j += 1
            ent_dist_counter = 0

            ent1_type_id = list(self.cdb.cui2type_ids.get(ent1._.cui, ''))
            ent1_types = [self.cdb.addl_info['type_id2name'].get(tui, '') for tui in ent1_type_id]

            for ent2 in doc.ents[j:]:
                if ent1.text.lower() != ent2.text.lower() and self.window_size and 1 < ent2.start - ent1.start <= self.window_size and ent_dist_counter < self.ent_context_left:
                    
                    ent2_type_id = list(self.cdb.cui2type_ids.get(ent2._.cui, ''))
                    ent2_types = [self.cdb.addl_info['type_id2name'].get(tui, '') for tui in ent2_type_id]

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

                    if self.window_size < (right_sentence_context - left_sentence_context):
                        continue

                    sentence_window_tokens = []

                    for token in doc[left_sentence_context:right_sentence_context]:
                        text = token.text.strip()
                        if text != "" and not any(chr in chars_to_exclude for chr in text):
                            sentence_window_tokens.append(text)

                    sentence_token_span = (sentence_window_tokens,
                                     (ent1.start - left_sentence_context, ent1.end - left_sentence_context),
                                     (ent2.start - left_sentence_context, ent2.end - left_sentence_context)
                       )

                    relation_instances.append([sentence_token_span, ent1.text, ent2.text, "UNK", self.blank_label_id, ent1_types, ent2_types, ent1._.id, ent2._.id, ent1._.cui, ent2._.cui, doc_id])
                    ent_dist_counter += 1

        return relation_instances
      
    def create_relations_from_export(self, data : Dict):
        """  
            Args:
                data (Dict):
                    MedCAT Export data.
            Returns:
                relations (List[list]) :
                   data columns: ["relation_token_span_ids", "ent1_ent2_start", "ent1", "ent2", "label", "label_id", "ent1_type", "ent2_type", "ent1_id", "ent2_id", "ent1_cui", "ent2_cui", "doc_id"]
                n_classes (int):
                   no. of classes detected
        """        

        output_relations = []

        punct_symbols = ['?', '.', ',', ';', ':', '#', '-', ]

        relation_type_filter_pairs = self.config.general["relation_type_filter_pairs"]

        for project in data['projects']:
            for doc_id, document in enumerate(project['documents']):
                text = str(document['text']) 
                if len(text) > 0:
                    annotations = document['annotations']
                    relations = document['relations']

                    if self.config.general['lowercase']:
                        text = text.lower()

                    doc_length = len(text)

                    tokenizer_data = self.tokenizer(text)
                    
                    ann_ids_ents = {}
                    for ann in annotations: 
                        ann_id = ann['id']
                        ann_ids_ents[ann_id] = {}
                        ann_ids_ents[ann_id]['cui'] = ann['cui']
                        ann_ids_ents[ann_id]['type_ids'] = list(self.cdb.cui2type_ids.get(ann['cui'], ''))
                      
                        ann_ids_ents[ann_id]['types'] = [self.cdb.addl_info['type_id2name'].get(tui, '') for tui in ann_ids_ents[ann_id]['type_ids']]

                    relation_instances = []
                    ann_ids_from_reliations = []

                    for relation in relations:
                        ann_start_start_pos = relation['start_entity_start_idx']
                        ann_start_end_pos = relation['start_entity_end_idx']
                        ann_end_start_pos = relation['end_entity_start_idx']
                        ann_end_end_pos = relation['end_entity_end_idx']

                        start_entity_value = relation['start_entity_value']
                        end_entity_value = relation['end_entity_value']

                        start_entity_id = relation['start_entity']
                        end_entity_id = relation['end_entity']
                        
                        start_entity_types = ann_ids_ents[start_entity_id]['types']
                        end_entity_types = ann_ids_ents[end_entity_id]['types']

                        for ent1type, ent2type in enumerate(relation_type_filter_pairs):
                            if ent1type not in start_entity_types and ent2type not in end_entity_types:
                                continue
                
                        start_entity_cui = ann_ids_ents[start_entity_id]['cui']
                        end_entity_cui = ann_ids_ents[end_entity_id]['cui']

                        ann_ids_from_reliations.extend([start_entity_id, end_entity_id])

                        relation_label = relation['relation']

                        if start_entity_id != end_entity_id and relation.get('validated', True):
                            
                            sent_start_pos = ann_start_start_pos
                            is_char_punctuation = False
     
                            while not is_char_punctuation and sent_start_pos >= 0 :
                                is_char_punctuation = text[sent_start_pos] in punct_symbols
                                sent_start_pos -= 1
                            
                            left_sentence_context  = sent_start_pos + 1 if sent_start_pos > 0 else 0
                            
                            sent_start_pos = ann_end_end_pos
                            is_char_punctuation = False
                            
                            while not is_char_punctuation and sent_start_pos < doc_length:
                                is_char_punctuation = text[sent_start_pos] in punct_symbols
                                sent_start_pos += 1
                            
                            right_sentence_context = sent_start_pos + 1 if sent_start_pos > 0 else doc_length
                            
                            start_input_idx = 0
                            end_input_idx = len(tokenizer_data["offset_mapping"]) - 1

                            for idx, pair in enumerate(tokenizer_data["offset_mapping"]):
                                if left_sentence_context >= pair[0] and left_sentence_context <= pair[1]:
                                    if start_input_idx == 0:
                                        start_input_idx = idx
                                if right_sentence_context >= pair[0] and right_sentence_context <= pair[1]:
                                    if end_input_idx == len(tokenizer_data["offset_mapping"]):
                                        end_input_idx = idx
                        
                            input_ids = tokenizer_data["input_ids"] # [start_input_idx:end_input_idx]
                            # for token in tokens_text:
                            #    if token in relation_section_text and token.strip() != "":
                            #        sentence_window_tokens.append(token.strip())

                            # sentence_token_span = (sentence_window_tokens, 
                            #                        (ann_start_start_pos - left_sentence_context, ann_start_end_pos),
                            #                        (ann_end_start_pos - left_sentence_context, ann_end_end_pos + right_sentence_context ) )

                            relation_instances.append([input_ids, (start_input_idx, end_input_idx), start_entity_value, end_entity_value , relation_label, self.blank_label_id,
                                                         start_entity_types, end_entity_types, start_entity_id, end_entity_id, start_entity_cui, end_entity_cui, doc_id])

                    output_relations.extend(relation_instances)

        all_relation_labels = [relation[4] for relation in output_relations]        
        
        n_classes, unique_labels, labels2idx, idx2label = self.get_labels(all_relation_labels)

        # replace label_id with actual detected label number
        for idx in range(len(output_relations)):
            output_relations[idx][5] = labels2idx[output_relations[idx][4]]

        return { "output_relations": output_relations, "n_classes": n_classes, "unique_labels": unique_labels, "labels2idx": labels2idx, "idx2label": idx2label}
    
    @staticmethod
    def get_labels(relation_labels : List[str]):
        labels2idx = {}
        idx2label = {}
        unique_labels = []
        n_classes = 0

        for relation_label in relation_labels:
            if relation_label not in labels2idx.keys():
                labels2idx[relation_label] = n_classes
                idx2label[n_classes] = relation_label
                n_classes += 1

        unique_labels = set(relation_labels)

        return n_classes, unique_labels, labels2idx, idx2label

    def __len__(self):
        return len(self.dataset['output_relations'])

    def __getitem__(self, idx):
        '''
            Args:
                idx (int):
                 index of item in the dataset dict
            Return:
                long tensors of the following the columns : input_ids 
        '''
        return torch.LongTensor(self.dataset['output_relations'][idx][0]),\
                torch.LongTensor(self.dataset['output_relations'][idx][1]),\
                torch.LongTensor([self.dataset['output_relations'][idx][5]])
