import os
import logging
from transformers import PretrainedConfig

from medcat.config_rel_cat import ConfigRelCAT


logger = logging.getLogger(__name__)


class BaseConfig_RelationExtraction(PretrainedConfig):
    """ Base class for the RelCAT models
    """
    name = "base-config-relcat"

    def __init__(self, pretrained_model_name_or_path, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "relcat"
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.hf_model_config: PretrainedConfig = kwargs.get("model_config", PretrainedConfig())

    def to_dict(self):
        output = super().to_dict()
        output["model_type"] = self.model_type
        output["pretrained_model_name_or_path"] = self.pretrained_model_name_or_path
        output["model_config"] = self.hf_model_config
        return output

    def save(self, save_path: str):
        self.hf_model_config.to_json_file(
            os.path.join(save_path, "model_config.json"))

    @classmethod
    def load(cls, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, **kwargs) -> "BaseConfig_RelationExtraction":

        model_config_path = os.path.join(pretrained_model_name_or_path, "model_config.json")
        model_config = BaseConfig_RelationExtraction(pretrained_model_name_or_path=pretrained_model_name_or_path, relcat_config=relcat_config, **kwargs)

        if os.path.exists(model_config_path):
            if "modern-bert" in relcat_config.general.tokenizer_name or \
               "modern-bert" in relcat_config.general.model_name:
                from medcat.utils.relation_extraction.modernbert.config import ModernBertConfig_RelationExtraction
                model_config = ModernBertConfig_RelationExtraction.load(model_config_path, relcat_config=relcat_config, **kwargs)
            elif "bert" in relcat_config.general.tokenizer_name or \
               "bert" in relcat_config.general.model_name:
                from medcat.utils.relation_extraction.bert.config import BertConfig_RelationExtraction
                model_config = BertConfig_RelationExtraction.load(model_config_path, relcat_config=relcat_config, **kwargs)
            elif "llama" in relcat_config.general.tokenizer_name or \
               "llama" in relcat_config.general.model_name:
                from medcat.utils.relation_extraction.llama.config import LlamaConfig_RelationExtraction
                model_config = LlamaConfig_RelationExtraction.load(model_config_path, relcat_config=relcat_config, **kwargs)
        else:
            if pretrained_model_name_or_path:
                model_config.hf_model_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)
            else:
                model_config.hf_model_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=relcat_config.general.model_name, **kwargs)
            logger.info("Loaded config from : " + model_config_path)

        return model_config
