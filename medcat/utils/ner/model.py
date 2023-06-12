from typing import Any, Tuple, Union

from medcat.ner.transformers_ner import TransformersNER
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.config import Config


class NerModel:

    """The NER model.

    This wraps a CAT instance and simplifies its use as a
    NER model.

    It provies methods for creating one from a TrnasformersNER
    as well as loading from a model pack (along with some validation).

    It also exposes some useful parts of the CAT it wraps such as
    the config and the concept database.
    """

    def __init__(self, cat: CAT) -> None:
        self.cat = cat

    def train(self, json_path: Union[str, list, None], *args, **kwargs) -> Tuple[Any, Any, Any]:
        """Train the underlying transformers NER model.

        All the extra arguments are passed to the TransformersNER train method.

        Args:
            json_path (Union[str, list, None]): The JSON file path to read the training data from.

        Returns:
            Tuple[Any, Any, Any]: df, examples, dataset
        """
        return self.cat._addl_ner[0].train(json_path, *args, **kwargs)

    @property
    def config(self) -> Config:
        return self.cat.config

    @property
    def cdb(self) -> CDB:
        return self.cat.cdb

    @classmethod
    def create(cls, ner: TransformersNER) -> 'NerModel':
        """Create a NER model with a TransformersNER

        Args:
            ner (TransformersNER): The TransformersNER instance.

        Returns:
            NerModel: The resulting model 
        """
        cat = CAT(cdb=ner.cdb, addl_ner=ner)
        return cls(cat)

    @classmethod
    def load_model_pack(cls, model_pack_path: str) -> 'NerModel':
        """Load NER model from model pack.

        The method first wraps the loaded CAT instance.

        Args:
            model_pack_path (str): The model pack path.

        Returns:
            NerModel: The resulting DeI model.
        """
        cat = CAT.load_model_pack(model_pack_path)
        return cls(cat)
