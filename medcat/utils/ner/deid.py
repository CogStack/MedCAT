"""De-identification model.

This describes a wrapper on the regular CAT model.
The idea is to simplify the use of a DeId-specific model.

It tackles two use cases
1) Creation of a deid model
2) Loading and use of a deid model

I.e for use case 1:

Instead of:
cat = CAT(cdb=ner.cdb, addl_ner=ner)

You can use:
deid = DeIdModel.create(ner)


And for use case 2:

Instead of:
cat = CAT.load_model_pack(model_pack_path)
anon_text = deid_text(cat, text)

You can use:
deid = DeIdModel.load_model_pack(model_pack_path)
anon_text = deid.deid_text(text)

Or if/when structured output is desired:
deid = DeIdModel.load_model_pack(model_pack_path)
anon_doc = deid(text)  # the spacy document

The wrapper also exposes some CAT parts directly:
- config
- cdb
"""
from typing import Union, Tuple, Any

from medcat.cat import CAT
from medcat.utils.ner.model import NerModel

from medcat.utils.ner.helpers import _deid_text as deid_text


class DeIdModel(NerModel):
    """The DeID model.

    This wraps a CAT instance and simplifies its use as a
    de-identification model.

    It provies methods for creating one from a TransformersNER
    as well as loading from a model pack (along with some validation).

    It also exposes some useful parts of the CAT it wraps such as
    the config and the concept database.
    """

    def __init__(self, cat: CAT) -> None:
        self.cat = cat

    def train(self, json_path: Union[str, list, None],
              *args, **kwargs) -> Tuple[Any, Any, Any]:
        return super().train(json_path, *args, train_nr=0, **kwargs)  # type: ignore

    def deid_text(self, text: str, redact: bool = False) -> str:
        """Deidentify text and potentially redact information.

        Args:
            text (str): The text to deidentify.
            redact (bool): Whether to redact the information.

        Returns:
            str: The deidentified text.
        """
        return deid_text(self.cat, text, redact=redact)

    @classmethod
    def load_model_pack(cls, model_pack_path: str) -> 'DeIdModel':
        """Load DeId model from model pack.

        The method first loads the CAT instance.

        It then makes sure that the model pack corresponds to a
        valid DeId model.

        Args:
            model_pack_path (str): The model pack path.

        Raises:
            ValueError: If the model pack does not correspond to a DeId model.

        Returns:
            DeIdModel: The resulting DeI model.
        """
        ner_model = NerModel.load_model_pack(model_pack_path)
        cat = ner_model.cat
        if not cls._is_deid_model(cat):
            raise ValueError(
                f"The model saved at {model_pack_path} is not a deid model "
                f"({cls._get_reason_not_deid(cat)})")
        model = cls(ner_model.cat)
        return model

    @classmethod
    def _is_deid_model(cls, cat: CAT) -> bool:
        return not bool(cls._get_reason_not_deid(cat))

    @classmethod
    def _get_reason_not_deid(cls, cat: CAT) -> str:
        if cat.vocab is not None:
            return "Has vocab"
        if len(cat._addl_ner) != 1:
            return f"Incorrect number of addl_ner: {len(cat._addl_ner)}"
        return ""
