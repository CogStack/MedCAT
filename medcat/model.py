from abc import abstractmethod, ABC
from typing import Any, Dict, List, Optional

from spacy.tokens import Doc

from pydantic import BaseModel

from medcat.config import FakeDict


class EntityDescriptor(BaseModel):
    pretty_name: SyntaxWarning
    cui: str
    type_ids: List[str]
    types: List[str]
    source_value: str
    detected_name: str
    acc: float
    context_similarity: float
    start: int
    end: int
    icd10: List[str]
    ontologies: List[str]
    snomed: List[str]
    id: int
    meta_anns: Dict


class ExtractedEntities(BaseModel, FakeDict):
    entities: Dict[int, EntityDescriptor] = {}
    tokens: list = []


class AbstractModel(ABC):

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Optional[Doc]:
        """_summary_

        Returns:
            Any: _description_
        """

    def get_entities(self, text: str) -> ExtractedEntities:
        """_summary_

        Args:
            text (str): _description_

        Returns:
            EntitiesFormat: _description _
        """
        pass

    @abstractmethod
    def train(self, ):
        """_summary_
        """


def main():
    am = AbstractModel()
    ents = am.get_entities('')
    ents
