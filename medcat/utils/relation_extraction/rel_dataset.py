from ast import literal_eval
from typing import Any, Iterable, List, Dict, Tuple
from torch.utils.data import Dataset
from spacy.tokens import Doc
import pandas
import torch

from medcat.cdb import CDB
from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.meta_cat.data_utils import Span
from medcat.utils.relation_extraction.tokenizer import TokenizerWrapperBERT


class RelData(Dataset):

    def __init__(self, tokenizer: TokenizerWrapperBERT, config: ConfigRelCAT, cdb: CDB):
        self.cdb = cdb
        self.config = config
        self.tokenizer = tokenizer
        self.blank_label_id = self.config.general.padding_idx
        self.dataset: Dict[Any, Any] = {}
        self.window_size = self.config.general.window_size
        self.ent_context_left = self.config.general.cntx_left
        self.ent_context_right = self.config.general.cntx_right

    def generate_base_relations(self, docs: Iterable[Doc]) -> List[Any]:
        '''
            Generate relations from Spacy CAT docs,
        '''
        output_relations = []
        for doc_id, doc in enumerate(docs):
            output_relations.append(self.create_base_relations_from_doc(doc,
                                                                        doc_id=doc_id,))

        return output_relations

    def create_base_relations_from_csv(self, csv_path):
        # Assumes the columns are as follows ["relation_token_span_ids", "ent1_ent2_start", "ent1", "ent2", "label",
        # "label_id", "ent1_type", "ent2_type", "ent1_id", "ent2_id", "ent1_cui", "ent2_cui", "doc_id", "sents"],
        # last column is the actual source text

        df = pandas.read_csv(csv_path, index_col=False,
                             sep='\t', encoding='utf-8')

        tmp_col_rel_token_col = df.pop("relation_token_span_ids")

        df.insert(0, "relation_token_span_ids", tmp_col_rel_token_col)

        text_cols = ["sents", "text"]

        df["ent1_ent2_start"] =  df["ent1_ent2_start"].apply(
            lambda x: literal_eval(str(x)))

        for col in text_cols:
            if col in df.columns:
                
                out_rels = []
                for row_idx in range(len(df[col])):
                    _text = df.iloc[row_idx][col]
                    _ent1_ent2_start = df.iloc[row_idx]["ent1_ent2_start"]
                    out_rels.append(self.create_base_relations_from_doc(_text, doc_id=row_idx, ent1_ent2_tokens_char_start_pos=_ent1_ent2_start))
                
                df["relation_token_span_ids"] = [ rel["output_relations"][0] for rel in out_rels]
                df = df.drop(columns=col)
                break

        nclasses, labels2idx, idx2label = RelData.get_labels(
            df["label"], self.config)

        output_relations = df.values.tolist()

        print("No. of relations detected:", len(output_relations),
              " from : ", csv_path, "classes: ", str(idx2label))

        # replace/update label_id with actual detected label number
        for idx in range(len(output_relations)):
            output_relations[idx][5] = labels2idx[output_relations[idx][4]]

        return {"output_relations": output_relations, "nclasses": nclasses, "labels2idx": labels2idx, "idx2label": idx2label}

    def create_base_relations_from_doc(self, doc: Doc | str, doc_id: str, ent1_ent2_tokens_char_start_pos: List | Tuple = (-1, -1)) -> Dict:
        """  
            doc : SpacyDoc
            window_size : int, Character distance between any two entities start positions.
            Creates a list of tuples based on pairs of entities detected (relation, ent1, ent2) for one spacy document.

            Returns:
                relations (List[list]) :
                   data columns: [
                                  "relation_token_span_ids", "ent1_ent2_start", "ent1", "ent2", "label",
                                  "label_id", "ent1_type", "ent2_type", "ent1_id", "ent2_id",
                                  "ent1_cui", "ent2_cui", "doc_id"
                                 ]
        """
        relation_instances = []

        chars_to_exclude = ":!@#$%^&*()-+?_=.,;<>/[]{}"
        tokenizer_data = None

        if isinstance(doc, str):
            tokenizer_data = self.tokenizer(doc)
            doc_text = doc
        elif isinstance(doc, Doc):
            tokenizer_data = self.tokenizer(doc.text)
            doc_text = doc.text

        doc_length = len(tokenizer_data["offset_mapping"])

        if ent1_ent2_tokens_char_start_pos != (-1, -1):
            ent1_token_pos, ent2_token_pos = -1, -1
            for i in range(0, doc_length):
                if ent1_ent2_tokens_char_start_pos[0] in range(tokenizer_data["offset_mapping"][i][0],tokenizer_data["offset_mapping"][i][1] + 1):
                    ent1_token_pos = i
                if ent1_ent2_tokens_char_start_pos[1] in range(tokenizer_data["offset_mapping"][i][0],tokenizer_data["offset_mapping"][i][1] + 1):
                    ent2_token_pos = i
            
            ent1_start_char_pos, _ = tokenizer_data["offset_mapping"][ent1_token_pos]
            ent2_start_char_pos, _ = tokenizer_data["offset_mapping"][ent2_token_pos]

            if ent2_start_char_pos - ent1_start_char_pos <= self.window_size:
                ent1_left_ent_context_token_pos_end = ent1_token_pos - self.ent_context_left
                if ent1_left_ent_context_token_pos_end < 0:
                    ent1_left_ent_context_token_pos_end = 0

                ent2_right_ent_context_token_pos_end = ent2_token_pos + self.ent_context_right
                if ent2_right_ent_context_token_pos_end >= doc_length:
                    ent2_right_ent_context_token_pos_end = doc_length - 1

                left_context_start_char_pos = tokenizer_data["offset_mapping"][ent1_left_ent_context_token_pos_end][0]
                right_context_start_end_pos = tokenizer_data["offset_mapping"][ent2_right_ent_context_token_pos_end][1]

                ent1_token = tokenizer_data["tokens"][ent1_token_pos]
                ent2_token = tokenizer_data["tokens"][ent2_token_pos]
                
                window_tokenizer_data = self.tokenizer(doc_text[left_context_start_char_pos:right_context_start_end_pos])
                
                ent1token_new = [tkn for tkn in window_tokenizer_data["tokens"] if tkn in ent1_token][0]
                ent2token_new = [tkn for tkn in window_tokenizer_data["tokens"] if tkn in ent2_token][0]
        
                ent1_token_id = self.tokenizer.token_to_id(ent1token_new)
                ent2_token_id = self.tokenizer.token_to_id(ent2token_new)
            
                ent1_token_pos = [pos for pos, token_id in enumerate(
                    window_tokenizer_data["input_ids"]) if token_id == ent1_token_id][0]
                ent2_token_pos = [pos for pos, token_id in enumerate(
                    window_tokenizer_data["input_ids"]) if token_id == ent2_token_id][0]
                
                ent1_ent2_new_start = (ent1_token_pos, ent2_token_pos)
                en1_start, en1_end = window_tokenizer_data["offset_mapping"][ent1_token_pos]
                en2_start, en2_end = window_tokenizer_data["offset_mapping"][ent2_token_pos]

                relation_instances.append([window_tokenizer_data["input_ids"], ent1_ent2_new_start, ent1token_new, ent2token_new, "UNK", self.blank_label_id,
                                None,None, None,None, None, None, doc_id, "",
                                en1_start, en1_end, en2_start, en2_end])
        elif isinstance(doc, Doc):
            for ent1_idx in range(0, len(doc.ents)):
                ent1_token: Span = doc.ents[ent1_idx]

                if ent1_token not in chars_to_exclude:
                    ent1_type_id = list(self.cdb.cui2type_ids.get(ent1_token._.cui, ''))
                    ent1_types = [self.cdb.addl_info['type_id2name'].get(
                        tui, '') for tui in ent1_type_id]
                    
                    ent2pos = ent1_idx
                    ent1_start = ent1_token.start
                    
                    ent1_left_ent_context_token_pos_end = ent1_idx - self.ent_context_left
                    if ent1_left_ent_context_token_pos_end < 0:
                        ent1_left_ent_context_token_pos_end = 0

                    ent2_right_ent_context_token_pos_end = ent2pos + self.ent_context_right
                    if ent2_right_ent_context_token_pos_end >= doc_length:
                        ent2_right_ent_context_token_pos_end = doc_length - 1
                    
                    left_context_start_char_pos = doc.ents[ent1_left_ent_context_token_pos_end].start
                    
                    for ent2_idx in range(len(doc.ents[ent2pos:ent2_right_ent_context_token_pos_end])):
                        ent2pos += 1

                        ent2_token: Span = doc.ents[ent2_idx]

                        if ent2_token not in chars_to_exclude:

                            ent2_type_id = list(
                                self.cdb.cui2type_ids.get(ent2_token._.cui, ''))
                            ent2_types = [self.cdb.addl_info['type_id2name'].get(
                                tui, '') for tui in ent2_type_id]
                    
                            ent2_start = ent2_token.start
                            if ent2_token != ent1_token and 1 < (ent2_start - ent1_start) <= self.window_size:
                                 # update val
                                ent2_right_ent_context_token_pos_end = ent2pos + self.ent_context_right
                                if ent2_right_ent_context_token_pos_end >= doc_length:
                                    ent2_right_ent_context_token_pos_end = doc_length

                                right_context_start_end_pos = doc.ents[ent2_right_ent_context_token_pos_end].end

                                window_tokenizer_data = self.tokenizer(doc_text[left_context_start_char_pos:right_context_start_end_pos])

                                ent1token_new = [
                                    tkn for tkn in window_tokenizer_data["tokens"] if tkn in ent1_token][0]
                                ent2token_new = [
                                    tkn for tkn in window_tokenizer_data["tokens"] if tkn in ent2_token][0]
                        
                                ent1_token_id = self.tokenizer.token_to_id(ent1token_new)
                                ent2_token_id = self.tokenizer.token_to_id(ent2token_new)
                            
                                ent1_token_pos = [pos for pos, token_id in enumerate(
                                    window_tokenizer_data["input_ids"]) if token_id == ent1_token_id][0]
                                ent2_token_pos = [pos for pos, token_id in enumerate(
                                    window_tokenizer_data["input_ids"]) if token_id == ent2_token_id][0]
                                
                                ent1_ent2_new_start = (ent1_token_pos, ent2_token_pos)
                                en1_start, en1_end = window_tokenizer_data["offset_mapping"][ent1_token_pos]
                                en2_start, en2_end = window_tokenizer_data["offset_mapping"][ent2_token_pos]
                                
                                relation_instances.append([window_tokenizer_data["input_ids"], ent1_ent2_new_start, ent1token_new, ent2token_new, "UNK", self.blank_label_id,
                                ent1_types, ent2_types, ent1_token._.id, ent2_token._.id, ent1_token._.cui, ent2_token._.cui, doc_id, "",
                                en1_start, en1_end, en2_start, en2_end])
       
        return {"output_relations": relation_instances, "nclasses": self.blank_label_id, "labels2idx": {}, "idx2label": {}}

    def create_relations_from_export(self, data: Dict):
        """  
            Args:
                data (Dict):
                    MedCAT Export data.
            Returns:
                relations (List[list]) :
                   data columns: ["relation_token_span_ids", "ent1_ent2_start", "ent1", "ent2", "label", "label_id", "ent1_type", "ent2_type", "ent1_id", "ent2_id", "ent1_cui", "ent2_cui", "doc_id"]
                nclasses (int):
                   no. of classes detected
        """

        output_relations = []

        punct_symbols = ['?', '.', ',', ';', ':', '#', '-', ]

        relation_type_filter_pairs = self.config.general.relation_type_filter_pairs

        for project in data['projects']:
            for doc_id, document in enumerate(project['documents']):
                text = str(document['text'])
                if len(text) > 0:
                    annotations = document['annotations']
                    relations = document['relations']

                    if self.config.general.lowercase:
                        text = text.lower()

                    doc_length = len(text)

                    tokenizer_data = self.tokenizer(text)

                    ann_ids_ents: Dict[Any, Any] = {}
                    for ann in annotations:
                        ann_id = ann['id']
                        ann_ids_ents[ann_id] = {}
                        ann_ids_ents[ann_id]['cui'] = ann['cui']
                        ann_ids_ents[ann_id]['type_ids'] = list(
                            self.cdb.cui2type_ids.get(ann['cui'], ''))

                        ann_ids_ents[ann_id]['types'] = [self.cdb.addl_info['type_id2name'].get(
                            tui, '') for tui in ann_ids_ents[ann_id]['type_ids']]

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

                        ann_ids_from_reliations.extend(
                            [start_entity_id, end_entity_id])

                        relation_label = relation['relation']

                        if start_entity_id != end_entity_id and relation.get('validated', True):

                            sent_start_pos = ann_start_start_pos
                            is_char_punctuation = False

                            while not is_char_punctuation and sent_start_pos >= 0:
                                is_char_punctuation = text[sent_start_pos] in punct_symbols
                                sent_start_pos -= 1

                            left_sentence_context = sent_start_pos + 1 if sent_start_pos > 0 else 0

                            sent_start_pos = ann_end_end_pos
                            is_char_punctuation = False

                            while not is_char_punctuation and sent_start_pos < doc_length:
                                is_char_punctuation = text[sent_start_pos] in punct_symbols
                                sent_start_pos += 1

                            right_sentence_context = sent_start_pos + \
                                1 if sent_start_pos > 0 else doc_length

                            start_input_idx = 0
                            end_input_idx = len(
                                tokenizer_data["offset_mapping"]) - 1

                            for idx, pair in enumerate(tokenizer_data["offset_mapping"]):
                                if left_sentence_context >= pair[0] and left_sentence_context <= pair[1]:
                                    if start_input_idx == 0:
                                        start_input_idx = idx
                                if right_sentence_context >= pair[0] and right_sentence_context <= pair[1]:
                                    if end_input_idx == len(tokenizer_data["offset_mapping"]):
                                        end_input_idx = idx

                            # [start_input_idx:end_input_idx]
                            input_ids = tokenizer_data["input_ids"]

                            relation_instances.append([input_ids, (start_input_idx, end_input_idx), start_entity_value, end_entity_value, relation_label, self.blank_label_id,
                                                       start_entity_types, end_entity_types, start_entity_id, end_entity_id, start_entity_cui, end_entity_cui, doc_id, "",
                                                       ann_start_start_pos, ann_start_end_pos, ann_end_start_pos, ann_end_end_pos])

                    output_relations.extend(relation_instances)

        all_relation_labels = [relation[4] for relation in output_relations]

        nclasses, labels2idx, idx2label = self.get_labels(
            all_relation_labels, self.config)

        # replace label_id with actual detected label number
        for idx in range(len(output_relations)):
            output_relations[idx][5] = labels2idx[output_relations[idx][4]]

        return {"output_relations": output_relations, "nclasses": nclasses, "labels2idx": labels2idx, "idx2label": idx2label}

    @classmethod
    def get_labels(cls, relation_labels: List[str], config: ConfigRelCAT) -> Any:
        labels2idx: Dict[str, int] = {}
        idx2label: Dict[int, str] = {}
        class_ids = 0

        config_labels2idx = config.general.labels2idx

        if len(list(config_labels2idx.values())) > 0:
            class_ids = max(list(config_labels2idx.values())) + 1

        for relation_label in relation_labels:
            if relation_label not in labels2idx.keys() and relation_label not in config_labels2idx.keys():
                labels2idx[relation_label] = class_ids
                config_labels2idx[relation_label] = class_ids
                class_ids += 1

            if relation_label not in config_labels2idx.keys():
                labels2idx[relation_label] = class_ids
                config_labels2idx[relation_label] = labels2idx[relation_label]
                class_ids += 1

            if relation_label in config_labels2idx.keys() and relation_label not in labels2idx.keys():
                labels2idx[relation_label] = config_labels2idx[relation_label]

            idx2label[labels2idx[relation_label]] = relation_label

        config.general.labels2idx = config_labels2idx

        return len(labels2idx.keys()), labels2idx, idx2label,

    def __len__(self):
        return len(self.dataset['output_relations'])

    def __getitem__(self, idx):
        '''
            Args:
                idx (int):
                 index of item in the dataset dict
            Return:
                long tensors of the following the columns : input_ids, ent1&ent2 token idx, label_ids
        '''
        return torch.LongTensor(self.dataset['output_relations'][idx][0]),\
            torch.LongTensor(self.dataset['output_relations'][idx][1]),\
            torch.LongTensor([self.dataset['output_relations'][idx][5]])
