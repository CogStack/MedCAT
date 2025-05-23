import logging
import os
from typing import cast

from medcat.config_rel_cat import ConfigRelCAT
from medcat.utils.relation_extraction.config import BaseConfig_RelationExtraction
from transformers import BertConfig

logger = logging.getLogger(__name__)


class BertConfig_RelationExtraction(BaseConfig_RelationExtraction):
    """ Class for BertConfig
    """

    name = 'bert-config'
    pretrained_model_name_or_path = "bert-base-uncased"
    hf_model_config: BertConfig

    @classmethod
    def load(cls, pretrained_model_name_or_path: str, relcat_config: ConfigRelCAT, **kwargs)  -> "BertConfig_RelationExtraction":
        model_config = cls(pretrained_model_name_or_path, **kwargs)

        if pretrained_model_name_or_path and os.path.exists(pretrained_model_name_or_path):
            model_config.hf_model_config = cast(BertConfig, BertConfig.from_json_file(pretrained_model_name_or_path))
            logger.info("Loaded config from file: " + pretrained_model_name_or_path)
        else:
            relcat_config.general.model_name = cls.pretrained_model_name_or_path
            model_config.hf_model_config = cast(BertConfig, BertConfig.from_pretrained(
                pretrained_model_name_or_path=cls.pretrained_model_name_or_path, **kwargs))
            logger.info("Loaded config from pretrained: " + relcat_config.general.model_name)

        return model_config
