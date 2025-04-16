import logging
import os
from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.config import BaseConfig_RelationExtraction
from transformers import LlamaConfig

logger = logging.getLogger(__name__)


class LlamaConfig_RelationExtraction(BaseConfig_RelationExtraction):
    """ Class for LlamaConfig
    """

    name = 'llama-config'
    pretrained_model_name_or_path = "meta-llama/Llama-3.1-8B"

    @classmethod
    def load(cls, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, **kwargs) -> "LlamaConfig_RelationExtraction":
        model_config = cls(pretrained_model_name_or_path, **kwargs)
        model_config_path = os.path.join(pretrained_model_name_or_path, "model_config.json")

        if pretrained_model_name_or_path and os.path.exists(model_config_path):
            model_config = LlamaConfig.from_json_file(model_config_path)
            logger.info("Loaded config from file: " + model_config_path)
        else:
            relcat_config.general.model_name = cls.pretrained_model_name_or_path
            model_config = LlamaConfig.from_pretrained(
                pretrained_model_name_or_path=cls.pretrained_model_name_or_path, **kwargs)
            logger.info("Loaded config from pretrained: " + relcat_config.general.model_name)


        return model_config
