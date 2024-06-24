import logging
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertModel
from transformers.models.bert.configuration_bert import BertConfig
from medcat.config_rel_cat import ConfigRelCAT

from transformers.models.llama import LlamaModel, LlamaConfig
from typing import Any, Optional, Tuple
from medcat.utils.relation_extraction.ml_utils import create_dense_layers, get_annotation_schema_tag


class BertModel_RelationExtraction(nn.Module):
    """ BertModel class for RelCAT
    """

    name = "bertmodel_relcat"

    log = logging.getLogger(__name__)

    def __init__(self, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, model_config: BertConfig):
        """ Class to hold the BERT model + model_config

        Args:
            pretrained_model_name_or_path (str): path to load the model from,
                    this can be a HF model i.e: "bert-base-uncased", if left empty, it is normally assumed that a model is loaded from 'model.dat'
                    using the RelCAT.load() method. So if you are initializing/training a model from scratch be sure to base it on some model.            
            relcat_config (ConfigRelCAT): relcat config.
            model_config (BertConfig): HF bert config for model.
        """
        super(BertModel_RelationExtraction, self).__init__()

        self.relcat_config: ConfigRelCAT = relcat_config
        self.model_config: BertConfig = model_config

        self.bert_model:BertModel = BertModel(config=model_config)

        if pretrained_model_name_or_path != "":
            self.bert_model = BertModel.from_pretrained(pretrained_model_name_or_path, config=model_config)

        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.drop_out = nn.Dropout(self.model_config.hidden_dropout_prob)

        if self.relcat_config.general.task == "pretrain":
            self.activation = nn.Tanh()
            self.cls = BertPreTrainingHeads(self.model_config)

        # dense layers
        self.fc1, self.fc2, self.fc3 = create_dense_layers(self.relcat_config)

        self.log.info("RelCAT BertConfig: " + str(self.model_config))

    def output2logits(self, pooled_output: torch.Tensor, sequence_output: torch.Tensor, input_ids: torch.Tensor, e1_e2_start: torch.Tensor) -> torch.Tensor:
        """

        Args:
            pooled_output (torch.Tensor): embedding of the CLS token
            sequence_output (torch.Tensor): hidden states/embeddings for each token in the input text
            input_ids (torch.Tensor): input token ids.
            e1_e2_start (torch.Tensor): annotation tags token position

        Returns:
            torch.Tensor: classification probabilities for each token.
        """

        new_pooled_output = pooled_output

        if self.relcat_config.general.annotation_schema_tag_ids:
            annotation_schema_tag_ids_ = [self.relcat_config.general.annotation_schema_tag_ids[i:i + 2] for i in
                                        range(0, len(self.relcat_config.general.annotation_schema_tag_ids), 2)]
            seq_tags = []

            # for each pair of tags (e1,s1) and (e2,s2)
            for each_tags in annotation_schema_tag_ids_:
                seq_tags.append(get_annotation_schema_tag(
                    sequence_output, input_ids, each_tags))

            seq_tags = torch.stack(seq_tags, dim=0)

            new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
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

        x = self.drop_out(new_pooled_output)
        x = self.fc1(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        classification_logits = self.fc3(x)
        return classification_logits.to(self.relcat_config.general.device)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Any = None,
                head_mask: Any = None,
                encoder_hidden_states: Any = None,
                encoder_attention_mask: Any = None,
                Q: Any = None,
                e1_e2_start: Any = None,
                pooled_output: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:

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

        model_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_attention_mask)

        # (batch_size, sequence_length, hidden_size)
        sequence_output = model_output[0]
        pooled_output = model_output[1]

        classification_logits = self.output2logits(
            pooled_output, sequence_output, input_ids, e1_e2_start)

        return model_output, classification_logits.to(self.relcat_config.general.device)


class LlamaModel_RelationExtraction(nn.Module):
    """ LlamaModel class for RelCAT
    """

    name = "llamamodel_relcat"

    log = logging.getLogger(__name__)

    def __init__(self, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, model_config: LlamaConfig):
        """ Class to hold the Llama model + model_config

        Args:
            pretrained_model_name_or_path (str): path to load the model from,
                    this can be a HF model i.e: "bert-base-uncased", if left empty, it is normally assumed that a model is loaded from 'model.dat'
                    using the RelCAT.load() method. So if you are initializing/training a model from scratch be sure to base it on some model.            
            relcat_config (ConfigRelCAT): relcat config.
            model_config (LlamaConfig): HF bert config for model.
        """

        super(LlamaModel_RelationExtraction, self).__init__()

        self.relcat_config: ConfigRelCAT = relcat_config
        self.model_config = model_config

        self.llama_model: LlamaModel = LlamaModel(config=model_config)

        if pretrained_model_name_or_path != "":
            self.llama_model = LlamaModel.from_pretrained(pretrained_model_name_or_path, config=model_config, ignore_mismatched_sizes=True)

        for param in self.llama_model.parameters():
            param.requires_grad = False

        self.drop_out = nn.Dropout(self.relcat_config.model.dropout)

        # dense layers
        self.fc1, self.fc2, self.fc3 = create_dense_layers(self.relcat_config)

        # for pooled output
        self.llama_pooler = LlamaPooler(self.model_config.hidden_size)

        self.log.info("RelCAT LlamaConfig: " + str(self.model_config))

    def output2logits(self, pooled_output: torch.Tensor, sequence_output: torch.Tensor, input_ids: torch.Tensor, e1_e2_start: torch.Tensor) -> torch.Tensor:
        """

        Args:
            pooled_output (torch.Tensor): embedding of the CLS token
            sequence_output (torch.Tensor): hidden states/embeddings for each token in the input text
            input_ids (torch.Tensor): input token ids.
            e1_e2_start (torch.Tensor): annotation tags token position

        Returns:
            torch.Tensor: classification probabilities for each token.
        """

        new_pooled_output = pooled_output

        if self.relcat_config.general.annotation_schema_tag_ids:
            annotation_schema_tag_ids_ = [self.relcat_config.general.annotation_schema_tag_ids[i:i + 2] for i in
                                        range(0, len(self.relcat_config.general.annotation_schema_tag_ids), 2)]
            seq_tags = []

            # for each pair of tags (e1,s1) and (e2,s2)
            for each_tags in annotation_schema_tag_ids_:
                seq_tags.append(get_annotation_schema_tag(
                    sequence_output, input_ids, each_tags))

            seq_tags = torch.stack(seq_tags, dim=0)

            if self.relcat_config.model.llama_use_pooled_output:
                new_pooled_output = torch.cat((pooled_output, *seq_tags), dim=1)
            else:
                new_pooled_output = torch.cat((seq_tags[0, seq_tags[1]]), dim=1)
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

        x = self.drop_out(new_pooled_output)
        x = self.fc1(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        classification_logits = self.fc3(x)
        return classification_logits.to(self.relcat_config.general.device)


    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                output_hidden_states: Optional[bool] = True,
                position_ids: Any = None,
                head_mask: Any = None,
                encoder_hidden_states: Any = None,
                encoder_attention_mask: Any = None,
                Q: Any = None,
                e1_e2_start: Any = None,
                pooled_output: Any = None) -> Tuple[torch.Tensor, torch.Tensor]:

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

        self.llama_model = self.llama_model.to(self.relcat_config.general.device)

        model_output = self.llama_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # (batch_size, sequence_length, hidden_size)
        sequence_output = model_output.last_hidden_state

        if self.relcat_config.model.llama_use_pooled_output:
            pooled_output = self.llama_pooler(model_output)
            pooled_output = pooled_output.to(self.relcat_config.general.device)
        else:
            pooled_output = model_output

        classification_logits = self.output2logits(
            pooled_output, sequence_output, input_ids, e1_e2_start)

        return model_output, classification_logits.to(self.relcat_config.general.device)


class LlamaPooler(nn.Module):
    """ An attempt to copy the BERT pooling technique for an increase in performance.

    Args:
        nn (nn.Module): .
    """
    def __init__(self, hidden_size: int):
        """ Initialises the pooler with a linear layer of size:
                self.model_config.hidden_size x self.model_config.hidden_size

        Args:
            hidden_size (int): size of tensor
        """
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. (original BERT)
        # We can do the same here but the [CLS] token equivalent is not the same as 
        # for bert as there is not much learning contained in it.
        # e.g: first_token_tensor = hidden_states[:, 0] # original
        # so instead we pool across all the tokens from the last hidden layer.

        pooled_output, _ = torch.max(hidden_states[-1], dim=1)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)

        return pooled_output
