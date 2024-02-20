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
from typing import Union, Tuple, Any, List, Iterable, Optional
import re
from medcat.cat import CAT
import warnings
from medcat.utils.ner.model import NerModel
from medcat.config_transformers_ner import ConfigTransformersNER
from medcat.utils.ner.helpers import _deid_text as deid_text, replace_entities_in_text

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

    def get_chunks(self, text:str, de_id_pipe,maximum_tkns) -> List[List[Any]]:

        blocks = [[]]
        if 'Roberta' in de_id_pipe.ner_pipe.tokenizer.__class__.__name__:

            whitespace_pattern = re.compile(r'\s')
            tok_output = de_id_pipe.ner_pipe.tokenizer(text, return_offsets_mapping=True)
            for token, (start, end) in zip(tok_output['input_ids'], tok_output['offset_mapping']):
                if token in [0, 2]:
                    continue
                if len(blocks[-1]) == maximum_tkns:
                    to_append_block = []
                    idx_chunk = -1
                    for i in range(len(blocks[-1]) - 1, len(blocks[-1]) - 21, -1):
                        # This method to avoid sub-word division is specific for RoBERTA's tokenizer (BPE)
                        if re.search(whitespace_pattern, de_id_pipe.ner_pipe.tokenizer.decode(blocks[-1][i][0],clean_up_tokenization_spaces=False)):
                            idx_chunk = i
                            break

                    if idx_chunk != -1:
                        to_append_block.extend(blocks[-1][i:])
                        del blocks[-1][idx_chunk:]

                    to_append_block.append((token, (start, end)))
                    blocks.append(to_append_block)

                else:
                    blocks[-1].append((token, (start, end)))

        else:
            # print("***********WARNING:************\nChunking functionality is implemented for RoBERTa model. The model detected is not RoBERTa, chunking omitted.\nPII information MAY BE REVEALED.")
            warnings.warn("\n\nChunking functionality is implemented for RoBERTa models. The detected model is not RoBERTa, so chunking is omitted. Be cautious, as PII information MAY BE REVEALED.")
            tok_output = de_id_pipe.ner_pipe.tokenizer(text, return_offsets_mapping=True)
            for token, (start, end) in zip(tok_output['input_ids'], tok_output['offset_mapping']):
                if token in [0, 2]:
                    continue
                blocks[-1].append((token, (start, end)))

        return blocks

    def deid_text(self, text: str, redact: bool = False,config: Optional[ConfigTransformersNER] = None) -> str:
        """Deidentify text and potentially redact information.

        v2: Changed to address the limit of 512 tokens. Adding chunking (break down the document into mini-documents and then run the model)
        v3 planned: Add overlapping window instead of a straight cut

        Args:
            text (str): The text to deidentify.
            redact (bool): Whether to redact the information.

        Returns:
            str: The deidentified text.
        """

        if config is None:
            config = ConfigTransformersNER()
        maximum_tkns = min(510,config.general['maximum_tokens_model']) # This is specific to RoBERTa model, with a maximum token limit is 512.
        de_id_pipe = self.cat.pipe._nlp.get_pipe("deid")

        blocks = self.get_chunks(text,de_id_pipe,maximum_tkns)
        anon_text = []
        for block in blocks:
            this_text = text[block[0][-1][0]:block[-1][-1][-1]]
            anon_ = deid_text(self.cat, this_text, redact=redact)
            anon_text.append(anon_)

        return " ".join(anon_text)

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

        Returns:
            List[str]: List of deidentified documents.
        """
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
