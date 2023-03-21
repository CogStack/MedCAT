import dill
from typing import Optional, Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TransformersTokenizerNER(object):
    """Args:
        hf_tokenizer
            Must be able to return token offsets.
        max_len:
            Max sequence length, if longer it will be split into multiple examples.
        id2type:
            Can be ignored in most cases, should be a map from token to 'start' or 'sub' meaning is the token
                a subword or the start/full word. For BERT 'start' is everything that does not begin with ##.
        cui2name:
            Map from CUI to full name for labels.
    """

    def __init__(self,
                 hf_tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 max_len: int = 512,
                 id2type: Optional[Dict] = None,
                 cui2name: Optional[Dict] = None) -> None:
        self.hf_tokenizer = hf_tokenizer
        self.max_len = max_len
        self.label_map = {'O': 0, 'X': 1} # We'll keep the 'X' in case id2type is provided
        self.id2type = id2type
        self.cui2name = cui2name

    def calculate_label_map(self, dataset) -> None:
        for cuis in dataset['ent_cuis']:
            for cui in cuis:
                if cui not in self.label_map:
                    self.label_map[cui] = len(self.label_map)

    def encode(self, examples: Dict, ignore_subwords: bool = False) -> Dict:
        """Used with huggingface datasets map function to convert medcat_ner dataset into the
        appropriate form for NER with BERT. It will split long text segments into max_len sequences.

        Args:
            examples:
                Stream of examples.
            ignore_subwords:
                If set to `True` subwords of any token will get the special label `X`.
        """
        self.hf_tokenizer = self.ensure_tokenizer()
        old_ids = examples['id']
        old_names = examples['name']
        examples['input_ids'] = []
        examples['labels'] = []
        examples['id'] = []
        examples['name'] = []

        for _ind, example in enumerate(zip(examples['text'], examples['ent_starts'],
                examples['ent_ends'], examples['ent_cuis'])):
            tokens = self.hf_tokenizer(example[0], return_offsets_mapping=True, add_special_tokens=False)
            entities = [(start, end, cui) for start, end, cui in zip(example[1],
                        example[2], example[3])]
            entities.sort(key=lambda x: x[0])
            input_ids = []
            labels = []

            tkn_part_of_entity = False
            for ind in range(len(tokens['offset_mapping'])):
                offset = tokens['offset_mapping'][ind]
                input_ids.append(tokens['input_ids'][ind])

                if entities and (offset[0] >= entities[0][0] and offset[1] <= entities[0][1]):
                    # Means this token is part of entity at position 0
                    tkn_part_of_entity = True
                    if not ignore_subwords or (self.id2type is not None and self.id2type[tokens['input_ids'][ind]] == 'start'):
                        labels.append(self.label_map[entities[0][2]])
                    else:
                        labels.append(self.label_map['X'])

                    if entities[0][1] <= offset[1]:
                        # If it is the last token of the entity, remove the entity as it is done
                        del entities[0]
                        tkn_part_of_entity = False # Set this so the next token is not removed

                else:
                    if tkn_part_of_entity:
                        del entities[0]
                        tkn_part_of_entity = False

                    if not ignore_subwords or (self.id2type is not None and self.id2type[tokens['input_ids'][ind]] == 'start'):
                        labels.append(self.label_map["O"])
                    else:
                        labels.append(self.label_map['X'])

                if len(input_ids) >= self.max_len:
                    # Split into multiple examples if too long
                    examples['input_ids'].append(input_ids)
                    examples['labels'].append(labels)
                    examples['id'].append(old_ids[_ind])
                    examples['name'].append(old_names[_ind])

                    input_ids = []
                    labels = []

            if input_ids:
                examples['input_ids'].append(input_ids)
                examples['labels'].append(labels)
                examples['id'].append(old_ids[_ind])
                examples['name'].append(old_names[_ind])

        return examples

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            dill.dump(self.__dict__, f)

    def ensure_tokenizer(self) -> PreTrainedTokenizerBase:
        if self.hf_tokenizer is None:
            raise ValueError("The tokenizer is not loaded yet")
        return self.hf_tokenizer

    @classmethod
    def load(cls, path: str) -> 'TransformersTokenizerNER':
        tokenizer = cls()
        with open(path, 'rb') as f:
            d = dill.load(f)
            for k in tokenizer.__dict__:
                if k in d:
                    tokenizer.__dict__[k] = d[k]
        return tokenizer
