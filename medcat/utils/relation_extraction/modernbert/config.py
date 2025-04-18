import logging
import os
from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.config import BaseConfig_RelationExtraction
from transformers import ModernBertConfig
from transformers import PretrainedConfig

logger = logging.getLogger(__name__)


class ModernBertConfig_RelationExtraction(BaseConfig_RelationExtraction):
    """ Class for ModernBertConfig
    """

    name = 'modern-bert-config'
    pretrained_model_name_or_path = "answerdotai/ModernBERT-base"

    @classmethod
    def load(cls, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, **kwargs) -> "ModernBertConfig_RelationExtraction":
        model_config = cls(pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs)
        model_config_path = os.path.join(pretrained_model_name_or_path, "model_config.json")


        if pretrained_model_name_or_path and os.path.exists(model_config_path):
            model_config.model_config = ModernBertConfig.from_json_file(model_config_path)
            logger.info("Loaded config from file: " + model_config_path)
        else:
            relcat_config.general.model_name = cls.pretrained_model_name_or_path
            model_config.model_config = ModernBertConfig.from_pretrained(
                pretrained_model_name_or_path=cls.pretrained_model_name_or_path, **kwargs)
            logger.info("Loaded config from pretrained: " + relcat_config.general.model_name)

        return model_config
