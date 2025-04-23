import logging
from typing import Union
import torch
import os
from torch import nn
from transformers import ModernBertModel

from medcat.config_rel_cat import ConfigRelCAT
from transformers import PreTrainedModel
from medcat.utils.relation_extraction.ml_utils import create_dense_layers
from medcat.utils.relation_extraction.models import BaseModel_RelationExtraction
from medcat.utils.relation_extraction.config import BaseConfig_RelationExtraction
from medcat.utils.relation_extraction.modernbert.config import ModernBertConfig_RelationExtraction



class ModernBertModel_RelationExtraction(BaseModel_RelationExtraction):
    """ ModernBertModel class for RelCAT
    """

    name = "modernbertmodel_relcat"

    log = logging.getLogger(__name__)

    def __init__(self, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, model_config: Union[BaseConfig_RelationExtraction, ModernBertConfig_RelationExtraction]):
        """ Class to hold the ModernBERT model + model_config

        Args:
            pretrained_model_name_or_path (str): path to load the model from,
                    this can be a HF model i.e: "bert-base-uncased", if left empty, it is normally assumed that a model is loaded from 'model.dat'
                    using the RelCAT.load() method. So if you are initializing/training a model from scratch be sure to base it on some model.
            relcat_config (ConfigRelCAT): relcat config.
            model_config (Union[BaseConfig_RelationExtraction | ModernBertConfig_RelationExtraction]): HF bert config for model.
        """
        super(ModernBertModel_RelationExtraction, self).__init__(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                    relcat_config=relcat_config,
                                                    model_config=model_config)

        self.relcat_config: ConfigRelCAT = relcat_config
        self.model_config: Union[BaseConfig_RelationExtraction, ModernBertConfig_RelationExtraction] = model_config
        self.pretrained_model_name_or_path: str = pretrained_model_name_or_path

        self.hf_model: Union[ModernBertModel, PreTrainedModel] = ModernBertModel(config=model_config.hf_model_config)

        for param in self.hf_model.parameters(): # type: ignore
            if self.relcat_config.model.freeze_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.drop_out = nn.Dropout(self.relcat_config.model.dropout)

        # dense layers
        self.fc1, self.fc2, self.fc3 = create_dense_layers(self.relcat_config)

    @classmethod
    def load(cls, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, model_config: Union[BaseConfig_RelationExtraction, ModernBertConfig_RelationExtraction], **kwargs) -> "ModernBertModel_RelationExtraction":

        model = ModernBertModel_RelationExtraction(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                             relcat_config=relcat_config,
                                             model_config=model_config)

        model_path = os.path.join(pretrained_model_name_or_path, "model.dat")

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=relcat_config.general.device))
            cls.log.info("Loaded model from file: " + model_path)
        elif pretrained_model_name_or_path:
            model.hf_model = ModernBertModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path, config=model_config.hf_model_config, ignore_mismatched_sizes=True, **kwargs)
            cls.log.info("Loaded model from pretrained: " + pretrained_model_name_or_path)
        else:
            model.hf_model = ModernBertModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path, config=model_config.hf_model_config, ignore_mismatched_sizes=True, **kwargs)
            cls.log.info("Loaded model from pretrained: " + cls.pretrained_model_name_or_path)

        return model
