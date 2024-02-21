from medcat.config_transformers_ner import ConfigTransformersNER
from typing import Union, Tuple, Any, List, Iterable, Optional
import re
import logging

logger = logging.getLogger(__name__)

MAX_TOKENS_ROBERTA = 510

def get_chunks(text:str, tokenizer, config: ConfigTransformersNER = None) -> List[str]:
    """Chunking class for De_Id

    This utility is to be used with the De_Id wrapper to create chunks of input text.
    It provides methods for creating chunks based on the NER model used.

    """

    if config is None:
        config = ConfigTransformersNER()

    if 'Roberta' in tokenizer.__class__.__name__:
        return get_chunks_roberta(text, tokenizer, config)
    else:
        logger.warning(
            "Chunking functionality is implemented for RoBERTa models. The detected model is not RoBERTa, so chunking is omitted. Be cautious, as PII information MAY BE REVEALED.")
        return [text]

def get_chunks_roberta(self, text:str, tokenizer, config: ConfigTransformersNER) -> List[str]:
    """Create chunks from the given input text.

       Args:
           text (str): The text to deidentify.

       Returns:
           str: The deidentified text.
   """

    blocks: List[List[tuple]] = [[]]
    if config.general['maximum_tokens_model']>510:
        logger.warning(
            "Number of tokens per chunk is greater than the limit for RoBERTa model. Reverted back to 510 tokens per chunk")

    maximum_tkns = min(MAX_TOKENS_ROBERTA, config.general['maximum_tokens_model'])  # This is specific to RoBERTa model, with a maximum token limit is 512.
    whitespace_pattern = re.compile(r'\s')
    tok_output = tokenizer.encode_plus(text, return_offsets_mapping=True)
    for token, (start, end) in zip(tok_output['input_ids'], tok_output['offset_mapping']):
        if token in [0, 2]:
            continue
        if len(blocks[-1]) == maximum_tkns:
            to_append_block = []
            idx_chunk = -1
            for i in range(len(blocks[-1]) - 1, len(blocks[-1]) - 21, -1):
                # This method to avoid sub-word division is specific for RoBERTA's tokenizer (BPE)
                if re.search(whitespace_pattern, tokenizer.decode(blocks[-1][i][0],clean_up_tokenization_spaces=False)):
                    idx_chunk = i
                    break

            if idx_chunk != -1:
                to_append_block.extend(blocks[-1][i:])
                del blocks[-1][idx_chunk:]

            to_append_block.append((token, (start, end)))
            blocks.append(to_append_block)

        else:
            blocks[-1].append((token, (start, end)))

    chunked_text: List = []
    for block in blocks:
        this_text = text[block[0][-1][0]:block[-1][-1][-1]]
        chunked_text.append(this_text)

    return chunked_text
