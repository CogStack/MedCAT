from typing import Any, List, Tuple, Union, Optional

from spacy.tokens import Doc

from medcat.ner.transformers_ner import TransformersNER
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.config import Config


class NerModel:

    """The NER model.

    This wraps a CAT instance and simplifies its use as a
    NER model.

    It provies methods for creating one from a TransformersNER
    as well as loading from a model pack (along with some validation).

    It also exposes some useful parts of the CAT it wraps such as
    the config and the concept database.
    """

    def __init__(self, cat: CAT) -> None:
        self.cat = cat

    def train(self, json_path: Union[str, list, None], train_nr: int = 0,
              *args, **kwargs) -> Tuple[Any, Any, Any]:
        """Train the underlying transformers NER model.

        All the extra arguments are passed to the TransformersNER train method.

        Args:
            json_path (Union[str, list, None]): The JSON file path to read the training data from.
            train_nr (int, optional): The number of the NER object in cat._addl_train to train. Defaults to 0.

        Returns:
            Tuple[Any, Any, Any]: df, examples, dataset
        """
        return self.cat._addl_ner[train_nr].train(json_path, *args, **kwargs)

    def __call__(self, text: Optional[str], *args, **kwargs) -> Optional[Doc]:
        """Get the annotated document for text.

        Undefined arguments and keyword arguments get passed on to
        the equivalent `CAT` method.

        Args:
            text (Optional[str]): The input text.

        Returns:
            Optional[Doc]: The annotated document.
        """
        return self.cat(text, *args, **kwargs)

    def get_entities(self, text: str, *args, **kwargs) -> dict:
        """Gets the entities recognized within a given text.

        The output format is identical to `CAT.get_entities`.

        Undefined arguments and keyword arguments get passed on to
        CAT.get_entities.

        Args:
            text (str): The input text.

        Returns:
            dict: The output entities.
        """
        return self.cat.get_entities(text, *args, **kwargs)

    @property
    def config(self) -> Config:
        return self.cat.config

    @property
    def cdb(self) -> CDB:
        return self.cat.cdb

    @classmethod
    def create(cls, ner: Union[TransformersNER, List[TransformersNER]]) -> 'NerModel':
        """Create a NER model with a TransformersNER

        Args:
            ner (Union[TransformersNER, List[TransformersNER]]): The TransformersNER instance(s).

        Returns:
            NerModel: The resulting model 
        """
        # expecting all to have the same CDB
        cdb = ner.cdb if isinstance(ner, TransformersNER) else ner[0].cdb
        cat = CAT(cdb=cdb, addl_ner=ner)
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
