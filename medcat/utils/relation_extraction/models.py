import logging
import torch
from typing import Any, Optional, Tuple, Union, cast
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from medcat.config_rel_cat import ConfigRelCAT
from transformers.models.llama import LlamaModel
from transformers import BertModel
from transformers import ModernBertModel
from medcat.utils.relation_extraction.config import BaseConfig_RelationExtraction
from medcat.utils.relation_extraction.ml_utils import create_dense_layers, get_annotation_schema_tag


class BaseModelBluePrint_RelationExtraction(nn.Module):
    """ Base class for the RelCAT models
    """

    hf_model: PreTrainedModel
    relcat_config: ConfigRelCAT
    model_config: PretrainedConfig
    drop_out: nn.Dropout
    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear

    def __init__(self, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, model_config: Union[PretrainedConfig, BaseConfig_RelationExtraction]):
        """ Class to hold the HF model + model_config

        Args:
            pretrained_model_name_or_path (str): path to load the model from,
                    this can be a HF model i.e: "bert-base-uncased", if left empty, it is normally assumed that a model is loaded from 'model.dat'
                    using the RelCAT.load() method. So if you are initializing/training a model from scratch be sure to base it on some model.
            relcat_config (ConfigRelCAT): relcat config.
            model_config (PretrainedConfig): HF bert config for model.
        """
        super(BaseModelBluePrint_RelationExtraction, self).__init__()

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
            pooled_output: Any = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for the model

        Args:
            input_ids (torch.Tensor): input token ids. Defaults to None.
            attention_mask (torch.Tensor): attention mask for the input ids. Defaults to None.
            token_type_ids (torch.Tensor): token type ids for the input ids. Defaults to None.
            position_ids (Any): The position IDs. Defaults to None.
            head_mask (Any): The head mask. Defaults to None.
            encoder_hidden_states (Any): Encoder hidden states. Defaults to None.
            encoder_attention_mask (Any): Encoder attention mask. Defaults to None.
            Q (Any): Q. Defaults to None.
            e1_e2_start (Any): start and end indices for the entities in the input ids. Defaults to None.
            pooled_output (Any): The pooled output. Defaults to None.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor]]: logits for the relation classification task.
        """
        return None

    def output2logits(self, pooled_output: torch.Tensor, sequence_output: torch.Tensor, input_ids: torch.Tensor, e1_e2_start: torch.Tensor) -> Optional[torch.Tensor]:
        """ Convert the output of the model to logits

        Args:
            pooled_output (torch.Tensor): output of the pooled layer.
            sequence_output (torch.Tensor): output of the sequence layer.
            input_ids (torch.Tensor): input token ids.
            e1_e2_start (torch.Tensor): start and end indices for the entities in the input ids.

        Returns:
            logits (torch.Tensor): logits for the relation classification task.
        """
        return None


class BaseModel_RelationExtraction(BaseModelBluePrint_RelationExtraction):

    name = "basemodel_relcat"
    log = logging.getLogger(__name__)

    def __init__(self, relcat_config: ConfigRelCAT,
                 model_config: BaseConfig_RelationExtraction,
                 pretrained_model_name_or_path):
        super(BaseModel_RelationExtraction, self).__init__(pretrained_model_name_or_path=pretrained_model_name_or_path, 
                                                          relcat_config=relcat_config,
                                                          model_config=model_config)

        self.relcat_config: ConfigRelCAT = relcat_config
        self.model_config: BaseConfig_RelationExtraction = model_config
        self.hf_model: Union[ModernBertModel, BertModel, LlamaModel, PreTrainedModel] = PreTrainedModel(config=model_config.hf_model_config) # type: ignore
        self.pretrained_model_name_or_path: str = pretrained_model_name_or_path

        self._reinitialize_dense_and_frozen_layers(relcat_config=relcat_config)

        self.log.info("RelCAT model config: " + str(self.model_config.hf_model_config))

    def _reinitialize_dense_and_frozen_layers(self, relcat_config: ConfigRelCAT) -> None:
        """ Reinitialize the dense layers of the model

        Args:
            relcat_config (ConfigRelCAT): relcat config.
        """

        self.drop_out = nn.Dropout(relcat_config.model.dropout)
        self.fc1, self.fc2, self.fc3 = create_dense_layers(relcat_config)

        for param in self.hf_model.parameters(): # type: ignore
            if self.relcat_config.model.freeze_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True

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

        self.hf_model = self.hf_model.to(self.relcat_config.general.device) # type: ignore

        model_output = self.hf_model(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_attention_mask) # type: ignore

        # (batch_size, sequence_length, hidden_size)
        sequence_output = model_output[0]
        pooled_output = model_output[1]

        classification_logits = self.output2logits(
            pooled_output, sequence_output, input_ids, e1_e2_start)

        return model_output, classification_logits.to(self.relcat_config.general.device)


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

    @classmethod
    def load(cls, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, model_config: BaseConfig_RelationExtraction) -> "BaseModel_RelationExtraction":
        """ Load the model from the given path

        Args:
            pretrained_model_name_or_path (str): path to load the model from.
            relcat_config (ConfigRelCAT): relcat config.
            model_config (BaseConfig_RelationExtraction): The model-specific config.

        returns:
            BaseModel_RelationExtraction: The loaded model.
        """

        model = BaseModel_RelationExtraction(relcat_config=relcat_config, model_config=model_config, pretrained_model_name_or_path=pretrained_model_name_or_path)

        if "modern-bert" in relcat_config.general.tokenizer_name or \
             "modern-bert" in relcat_config.general.model_name:
            from medcat.utils.relation_extraction.modernbert.model import ModernBertModel_RelationExtraction
            from medcat.utils.relation_extraction.modernbert.config import ModernBertConfig_RelationExtraction
            model = ModernBertModel_RelationExtraction.load(pretrained_model_name_or_path, relcat_config=relcat_config,
                                                            model_config=cast(ModernBertConfig_RelationExtraction, model_config))
        elif "bert" in relcat_config.general.tokenizer_name or \
             "bert" in relcat_config.general.model_name:
            from medcat.utils.relation_extraction.bert.model import BertModel_RelationExtraction
            from medcat.utils.relation_extraction.bert.config import BertConfig_RelationExtraction
            model = BertModel_RelationExtraction.load(pretrained_model_name_or_path, relcat_config=relcat_config,
                                                      model_config=cast(BertConfig_RelationExtraction, model_config))
        elif "llama" in relcat_config.general.tokenizer_name or \
             "llama" in relcat_config.general.model_name:
            from medcat.utils.relation_extraction.llama.model import LlamaModel_RelationExtraction
            from medcat.utils.relation_extraction.llama.config import LlamaConfig_RelationExtraction
            model = LlamaModel_RelationExtraction.load(pretrained_model_name_or_path, relcat_config=relcat_config,
                                                       model_config=cast(LlamaConfig_RelationExtraction, model_config))
        else:
            if pretrained_model_name_or_path:
                model.hf_model = PreTrainedModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, config=model_config)
            else:
                model.hf_model = PreTrainedModel.from_pretrained(pretrained_model_name_or_path=relcat_config.general.model_name, config=model_config)
                cls.log.info("Loaded model from relcat_config: " + relcat_config.general.model_name)

        cls.log.info("Loaded " + str(model.__class__.__name__) + " from pretrained_model_name_or_path: " + pretrained_model_name_or_path)

        model._reinitialize_dense_and_frozen_layers(relcat_config=relcat_config)

        return model
