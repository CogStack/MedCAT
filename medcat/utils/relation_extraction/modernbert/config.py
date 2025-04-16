import logging
import os
from medcat.utils.relation_extraction.config import BaseConfig_RelationExtraction
from transformers import ModernBertConfig

logger = logging.getLogger(__name__)


class ModernBertConfig_RelationExtraction(BaseConfig_RelationExtraction):
    """ Class for ModernBertConfig
    """

    name = 'modern-bert-config'
    pretrained_model_name_or_path = "answerdotai/ModernBERT-base"

    @classmethod
    def load(cls, pretrained_model_name_or_path: str, **kwargs):
        model_config = cls(pretrained_model_name_or_path, **kwargs)
        model_config_path = os.path.join(pretrained_model_name_or_path, "model_config.json")

        if pretrained_model_name_or_path and os.path.exists(model_config_path):
            model_config = ModernBertConfig.from_json_file(model_config_path)
            logger.info("Loaded config from file: " + model_config_path)
        else:
            model_config = ModernBertConfig.from_pretrained(
                pretrained_model_name_or_path=cls.pretrained_model_name_or_path, **kwargs)
            logger.info("Loaded config from pretrained: " + cls.pretrained_model_name_or_path)

        return model_config
