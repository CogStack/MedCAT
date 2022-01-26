import torch
from torch import nn

from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertModel
from transformers.models.bert.configuration_bert import BertConfig

from transformers import logging


class BertModel_RelationExtraction(BertPreTrainedModel):
    def __init__(self, pretrained_model_name_or_path, model_config: BertConfig, model_size: str, task: str = "train", nclasses: int = 2, ignore_mismatched_sizes: bool = False):
        super(BertModel_RelationExtraction, self).__init__(model_config, ignore_mismatched_sizes)

        self.model_config = model_config
        self.nclasses = nclasses
        self.task = task
        self.model_size = model_size
        self.model= BertModel(model_config)
        self.drop_out = nn.Dropout(model_config.hidden_dropout_prob)

        self.hidden_size = self.model_config.hidden_size

        if self.task == "pretrain":
            self.activation = nn.Tanh()
            self.cls = BertPreTrainingHeads(self.model_config)

        elif self.task == "train":
            if self.model_size == 'bert-base-uncased':
                self.hidden_size = 2304
            elif self.model_size == 'bert-large-uncased':
                self.hidden_size = 3072
            elif 'biobert' in self.model_size:
                self.hidden_size = self.model_config.hidden_size * 3

        self.classification_layer = nn.Linear(self.hidden_size, self.nclasses)

        logging.set_verbosity_error()

        print("Model config: ", self.model_config)

        self.init_weights()


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                Q=None, e1_e2_start=None, pooled_output=None):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        self.classification_layer = nn.Linear(self.hidden_size, self.nclasses, device=device)

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        encoder_attention_mask = encoder_attention_mask.to(device)

        self.model = self.model.to(device)

        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask)

        sequence_output = model_output[0] # (batch_size,  sequence_length, hidden_size)
        pooled_output = model_output[1]

        e1e2_output =[]  
        temp_e1 = []
        temp_e2 = []

        for i, seq in enumerate(sequence_output): 
            # e1e2 token sequences
            temp_e1.append(seq[e1_e2_start[i][0]]) 
            temp_e2.append(seq[e1_e2_start[i][1]])

        e1e2_output.append(torch.stack(temp_e1, dim=0))
        e1e2_output.append(torch.stack(temp_e2, dim=0))

        new_pooled_output=torch.cat((pooled_output, *e1e2_output), dim=1)

        del e1e2_output
        del temp_e2
        del temp_e1

        classification_logits = None

        if self.task == "train":
            classification_logits = self.classification_layer(self.drop_out(new_pooled_output))

        return model_output, classification_logits.to(device)
