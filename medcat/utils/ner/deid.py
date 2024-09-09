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
from typing import Union, Tuple, Any, List, Iterable, Optional, Dict
import logging

from medcat.cat import CAT
from medcat.utils.ner.model import NerModel

from medcat.utils.ner.helpers import replace_entities_in_text


logger = logging.getLogger(__name__)


class DeIdModel(NerModel):
    """The DeID model.

    This wraps a CAT instance and simplifies its use as a
    de-identification model.

    It provides methods for creating one from a TransformersNER
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

        De-identified text.
        If redaction is enabled, identifiable entities will be
        replaced with starts (e.g `*****`).
        Otherwise, the replacement will be the CUI or in other words,
        the type of information that was hidden (e.g [PATIENT]).

        Args:
            text (str): The text to deidentify.
            redact (bool): Whether to redact the information.

        Returns:
            str: The deidentified text.
        """
        entities = self.cat.get_entities(text)['entities']
        return replace_entities_in_text(text, entities, self.cat.cdb.get_name, redact=redact)

    def deid_multi_texts(self,
                         texts: Union[Iterable[str], Iterable[Tuple]],
                         redact: bool = False,
                         addl_info: List[str] = ['cui2icd10', 'cui2ontologies', 'cui2snomed'],
                         n_process: Optional[int] = None,
                         batch_size: Optional[int] = None) -> List[str]:
        """Deidentify text on multiple branches

        Args:
            texts (Union[Iterable[str], Iterable[Tuple]]): Text to be annotated
            redact (bool): Whether to redact the information.
            addl_info (List[str], optional): Additional info. Defaults to ['cui2icd10', 'cui2ontologies', 'cui2snomed'].
            n_process (Optional[int], optional): Number of processes. Defaults to None.
            batch_size (Optional[int], optional): The size of a batch. Defaults to None.

        Raises:
            ValueError: In case of unsupported input.

        Returns:
            List[str]: List of deidentified documents.
        """
        # NOTE: we assume we're using the 1st (and generally only)
        #       additional NER model.
        #       the same assumption is made in the `train` method
        chunking_overlap_window = self.cat._addl_ner[0].config.general.chunking_overlap_window
        if chunking_overlap_window is not None:
            logger.warning("Chunking overlap window has been set to %s. "
                           "This may cause multiprocessing to stall in certain"
                           "environments and/or situations and has not been"
                           "fully tested.",
                           chunking_overlap_window)
            logger.warning("If the following hangs forever (i.e doesn't finish) "
                           "but you still wish to run on multiple processes you can set "
                           "`cat._addl_ner[0].config.general.chunking_overlap_window = None` "
                           "and then either a) save the model on disk and load it back up, or "
                           " b) call `cat._addl_ner[0].create_eval_pipeline()` to recreate the pipe. "
                           "However, this will remove chunking from the input text, which means "
                           "only the first 512 tokens will be recognised and thus only the "
                           "first part of longer documents (those with more than 512) tokens"
                           "will be deidentified. ")
        entities = self.cat.get_entities_multi_texts(texts, addl_info=addl_info,
                                                     n_process=n_process, batch_size=batch_size)
        out = []
        for raw_text, _ents in zip(texts, entities):
            ents = _ents['entities']
            text: str
            if isinstance(raw_text, tuple):
                text = raw_text[1]
            elif isinstance(raw_text, str):
                text = raw_text
            else:
                raise ValueError(f"Unknown raw text: {type(raw_text)}: {raw_text}")
            new_text = replace_entities_in_text(text, ents, get_cui_name=self.cat.cdb.get_name, redact=redact)
            out.append(new_text)
        return out

    @classmethod
    def load_model_pack(cls, model_pack_path: str, config: Optional[Dict] = None) -> 'DeIdModel':
        """Load DeId model from model pack.

        The method first loads the CAT instance.

        It then makes sure that the model pack corresponds to a
        valid DeId model.

        Args:
            config: Config for DeId model pack (primarily for stride of overlap window)
            model_pack_path (str): The model pack path.

        Raises:
            ValueError: If the model pack does not correspond to a DeId model.

        Returns:
            DeIdModel: The resulting DeI model.
        """
        ner_model = NerModel.load_model_pack(model_pack_path,config=config)
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
