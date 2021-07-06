class TokenizerNER(object):
    r'''

    Args:
        hf_tokenizer
            Must be able to return token offsets
        max_len:
            Max sequence length, if longer it will be split into multiple examples
    '''

    def __init__(self, hf_tokenizer, max_len=512, id2type=None):
        self.hf_tokenizer = hf_tokenizer
        self.max_len = max_len
        self.label_map = {'O': 0, 'X': 1}
        self.id2type = id2type

    def encode(self, examples, ignore_subwords=False):
        r''' Used with huggingface datasets map function to convert medcat_ner dataset into the
        appropriate form for NER with BERT. It will split long text segments into max_len sequences.

        Args:
            examples:
                Stream of examples
            ignore_subwords:
                If set to `True` subwords of any token will get the special label `X`
        '''
        old_ids = examples['id']
        examples['input_ids'] = []
        examples['labels'] = []
        examples['id'] = []

        for _ind, example in enumerate(zip(examples['text'], examples['ent_starts'],
                examples['ent_ends'], examples['ent_cuis'])):
            tokens = self.hf_tokenizer(example[0], return_offsets_mapping=True,
                    add_special_tokens=False)
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
                    if entities[0][2] not in self.label_map:
                        self.label_map[entities[0][2]] = len(self.label_map)
                    # Means this token is part of entity at position 0
                    tkn_part_of_entity = True
                    if not ignore_subwords or self.id2type[tokens['input_ids'][ind]] == 'start':
                        labels.append(self.label_map[entities[0][2]])
                    else:
                        labels.append(self.label_map['X'])

                else:
                    if tkn_part_of_entity:
                        del entities[0]
                        tkn_part_of_entity = False

                    if not ignore_subwords or self.id2type[tokens['input_ids'][ind]] == 'start':
                        labels.append(self.label_map["O"])
                    else:
                        labels.append(self.label_map['X'])

                if len(input_ids) >= self.max_len:
                    # Split into multiple examples if too long
                    examples['input_ids'].append(input_ids)
                    examples['labels'].append(labels)
                    examples['id'].append(old_ids[_ind])

                    input_ids = []
                    labels = []

            if input_ids:
                examples['input_ids'].append(input_ids)
                examples['labels'].append(labels)
                examples['id'].append(old_ids[_ind])

        return examples

