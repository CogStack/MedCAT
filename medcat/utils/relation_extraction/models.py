import torch
from torch import nn

from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertModel
from transformers.models.bert.configuration_bert import BertConfig

from medcat.config_rel_cat import ConfigRelCAT


class BertModel_RelationExtraction(BertPreTrainedModel):
    def __init__(self, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, model_config: BertConfig, ignore_mismatched_sizes: bool = False):
        super(BertModel_RelationExtraction, self).__init__(
            model_config, ignore_mismatched_sizes)

        self.relcat_config = relcat_config
        self.model_config = model_config
        self.bert_model = BertModel(model_config)
        self.drop_out = nn.Dropout(model_config.hidden_dropout_prob)

        if relcat_config.general.task == "pretrain":
            self.activation = nn.Tanh()
            self.cls = BertPreTrainingHeads(self.model_config)

        self.classification_layer = nn.Linear(self.relcat_config.model.model_size, relcat_config.train.nclasses)

        print("RelCAT Model config: ", self.model_config)

        self.init_weights()  # type: ignore

    def get_annotation_schema_tag(self, sequence_output, input_ids, special_tag):
        spec_idx = (input_ids == special_tag).nonzero(as_tuple=False)

        # remove duplicate positions (some datasets arent fully clean)
        initial_list = spec_idx[:, 1].tolist()

        pos = [(i, initial_list[i]) for i in range(len(initial_list))]
        pos_count = {}
        for i in range(len(pos)):
            if pos[i][0] not in list(pos_count.keys()):
                pos_count[pos[i][0]] = [1, [pos[i][0]]]
            else:
                pos_count[pos[i][0]][0] += 1
                pos_count[pos[i][0]][1].append(pos[i][0])

        dupe_pos = [i for i in range(
            len(initial_list)) if pos_count[pos[i][0]][0] != 1]

        for i in dupe_pos:
            spec_idx = torch.cat((spec_idx[:i], spec_idx[(i+1):]))

        temp = []
        for idx in spec_idx:
            temp.append(sequence_output[idx[0]][idx[1]])

        tags_rep = torch.stack(temp, dim=0)

        return tags_rep

    def output2logits(self, pooled_output, sequence_output, input_ids, e1_e2_start):
        """

        Args:
            data_iterator (Iterable):
                Simple iterator over sentences/documents, e.g. a open file
                or an array or anything that we can use in a for loop
                If True resume the previous training; If False, start a fresh new training.
        """

        new_pooled_output = pooled_output

        if self.relcat_config.general.annotation_schema_tag_ids:
            seq_tags = []
            for each_tag in self.relcat_config.general.annotation_schema_tag_ids:
                seq_tags.append(self.get_annotation_schema_tag(
                    sequence_output, input_ids, each_tag))
                
           # print(seq_tags)

            seq_tags = torch.stack(seq_tags, dim=0)

            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
            new_pooled_output = torch.squeeze(new_pooled_output, dim=1)
        else:
            e1e2_output = []
            temp_e1 = []
            temp_e2 = []

            for i, seq in enumerate(sequence_output):
                # e1e2 token sequences
                temp_e1.append(seq[e1_e2_start[i][0]])
                temp_e2.append(seq[e1_e2_start[i][1]])

            e1e2_output.append(torch.stack(temp_e1, dim=0))
            e1e2_output.append(torch.stack(temp_e2, dim=0))

            new_pooled_output = torch.cat((pooled_output, *e1e2_output), dim=1)

            del e1e2_output
            del temp_e2
            del temp_e1

        classification_logits = self.classification_layer(
            self.drop_out(new_pooled_output))

        return classification_logits.to(self.relcat_config.general.device)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                Q=None, e1_e2_start=None, pooled_output=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        if attention_mask is None:
            attention_mask = torch.ones(
                input_shape, device=self.relcat_config.general.device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                input_shape, device=self.relcat_config.general.device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.relcat_config.general.device)

        input_ids = input_ids.to(self.relcat_config.general.device)
        attention_mask = attention_mask.to(self.relcat_config.general.device)
        encoder_attention_mask = encoder_attention_mask.to(
            self.relcat_config.general.device)

        self.bert_model = self.bert_model.to(self.relcat_config.general.device)

        model_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_attention_mask)

        # (batch_size, sequence_length, hidden_size)
        sequence_output = model_output[0]
        pooled_output = model_output[1]

        classification_logits = self.output2logits(
            pooled_output, sequence_output, input_ids, e1_e2_start)

        return model_output, classification_logits.to(self.relcat_config.general.device)
