from typing import Union
from torch import nn
from transformers.models.bert.modeling_bert import BertModel
from transformers import ModernBertModel
from transformers.models.llama import LlamaModel


class Base_RelationExtraction(nn.Module):

    hf_model: Union[BertModel, ModernBertModel, LlamaModel]
