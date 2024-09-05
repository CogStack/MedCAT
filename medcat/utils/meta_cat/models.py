import torch
from collections import OrderedDict
from typing import Optional, Any, List, Iterable
from torch import nn, Tensor
from transformers import BertModel, AutoConfig
from medcat.meta_cat import ConfigMetaCAT
import logging
logger = logging.getLogger(__name__)


class LSTM(nn.Module):
    def __init__(self, embeddings: Optional[Tensor], config: ConfigMetaCAT) -> None:
        super(LSTM, self).__init__()

        self.config = config
        # Get the required sizes
        vocab_size = config.general['vocab_size']
        embedding_size = config.model['input_size']

        # Initialize embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=config.model['padding_idx'])
        if embeddings is not None:
            self.embeddings.load_state_dict(OrderedDict([('weight', embeddings)]))
            # Disable training for the embeddings - IMPORTANT
            self.embeddings.weight.requires_grad = config.model['emb_grad']

        # Create the RNN cell - divide
        self.rnn = nn.LSTM(input_size=config.model['input_size'],
                           hidden_size=config.model['hidden_size'] // config.model['num_directions'],
                           num_layers=config.model['num_layers'],
                           dropout=config.model['dropout'],
                           bidirectional=config.model['num_directions'] == 2)
        self.fc1 = nn.Linear(config.model['hidden_size'], config.model['nclasses'])

        self.d1 = nn.Dropout(config.model['dropout'])

    def forward(self,
                input_ids: torch.LongTensor,
                center_positions: Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                ignore_cpos: bool = False) -> Tensor:
        x = input_ids
        # Get the mask from x
        if attention_mask is None:
            mask = x != self.config.model['padding_idx']
        else:
            mask = attention_mask

        # Embed the input: from id -> vec
        x = self.embeddings(x)  # x.shape = batch_size x sequence_length x emb_size

        # Tell RNN to ignore padding and set the batch_first to True
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int().view(-1).cpu(), batch_first=True,
                                              enforce_sorted=False)

        # Run 'x' through the RNN
        x, hidden = self.rnn(x)

        # Add the padding again
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Get what we need
        # row_indices = torch.arange(0, x.size(0)).long()

        # If this is  True we will always take the last state and not CPOS
        if ignore_cpos:
            x = hidden[0]
            x = x.view(self.config.model['num_layers'], self.config.model['num_directions'], -1,
                       self.config.model['hidden_size'] // self.config.model['num_directions'])
            x = x[-1, :, :, :].permute(1, 2, 0).reshape(-1, self.config.model['hidden_size'])
        else:
            x_all = []
            for i, indices in enumerate(center_positions):
                this_hidden = x[i, indices, :]
                to_append, _ = torch.max(this_hidden, dim=0)
                x_all.append(to_append)

            x = torch.stack(x_all)

        # Push x through the fc network and add dropout
        x = self.d1(x)
        x = self.fc1(x)

        return x


class BertForMetaAnnotation(nn.Module):
    _keys_to_ignore_on_load_unexpected: List[str] = [r"pooler"]  # type: ignore

    def __init__(self, config):
        super(BertForMetaAnnotation, self).__init__()
        _bertconfig = AutoConfig.from_pretrained(config.model.model_variant,num_hidden_layers=config.model['num_layers'])
        if config.model['input_size'] != _bertconfig.hidden_size:
            logger.warning("Input size for %s model should be %d, provided input size is %d. Input size changed to %d",config.model.model_variant,_bertconfig.hidden_size,config.model['input_size'],_bertconfig.hidden_size)

        bert = BertModel.from_pretrained(config.model.model_variant, config=_bertconfig)
        self.config = config
        self.config.use_return_dict = False
        self.bert = bert
        self.num_labels = config.model["nclasses"]
        for param in self.bert.parameters():
            param.requires_grad = not config.model.model_freeze_layers

        hidden_size_2 = int(config.model.hidden_size / 2)
        # dropout layer
        self.dropout = nn.Dropout(config.model.dropout)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(_bertconfig.hidden_size*2, config.model.hidden_size)
        # dense layer 2
        self.fc2 = nn.Linear(config.model.hidden_size, hidden_size_2)
        # dense layer 3
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_2)
        # dense layer 3 (Output layer)
        model_arch_config = config.model.model_architecture_config

        if model_arch_config['fc3'] is True and model_arch_config['fc2'] is False:
            logger.warning("FC3 can only be used if FC2 is also enabled. Enabling FC2...")
            config.model.model_architecture_config['fc2'] = True

        if model_arch_config is not None:
            if model_arch_config['fc2'] is True:
                self.fc4 = nn.Linear(hidden_size_2, self.num_labels)
            else:
                self.fc4 = nn.Linear(config.model.hidden_size, self.num_labels)
        else:
            self.fc4 = nn.Linear(hidden_size_2, self.num_labels)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            center_positions: Iterable[Any] = [],
            ignore_cpos: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None
    ):
        """labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.

        Args:
            input_ids (Optional[torch.LongTensor]): The input IDs. Defaults to None.
            attention_mask (Optional[torch.FloatTensor]): The attention mask. Defaults to None.
            token_type_ids (Optional[torch.LongTensor]): Type IDs of the tokens. Defaults to None.
            position_ids (Optional[torch.LongTensor]): Position IDs. Defaults to None.
            head_mask (Optional[torch.FloatTensor]): Head mask. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor]): Input embeddings. Defaults to None.
            labels (Optional[torch.LongTensor]): Labels. Defaults to None.
            center_positions (Optional[Any]): Cennter positions. Defaults to None.
            output_attentions (Optional[bool]): Output attentions. Defaults to None.
            ignore_cpos: If center positions are to be ignored.
            output_hidden_states (Optional[bool]): Output hidden states. Defaults to None.
            return_dict (Optional[bool]): Whether to return a dict. Defaults to None.

        Returns:
            TokenClassifierOutput: The token classifier output.
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict # type: ignore

        outputs = self.bert(  # type: ignore
            input_ids,
            attention_mask=attention_mask, output_hidden_states=True
        )

        x_all = []
        for i, indices in enumerate(center_positions):
            this_hidden: torch.Tensor = outputs.last_hidden_state[i, indices, :]
            to_append, _ = torch.max(this_hidden, dim=0)
            x_all.append(to_append)

        x = torch.stack(x_all)

        pooled_output = outputs[1]
        x = torch.cat((x, pooled_output), dim=1)

        # fc1
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)

        if self.config.model.model_architecture_config is not None:
            if self.config.model.model_architecture_config['fc2'] is True:
                # fc2
                x = self.fc2(x)
                x = self.relu(x)
                x = self.dropout(x)

                if self.config.model.model_architecture_config['fc3'] is True:
                    # fc3
                    x = self.fc3(x)
                    x = self.relu(x)
                    x = self.dropout(x)
        else:
            # fc2
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)

            # fc3
            x = self.fc3(x)
            x = self.relu(x)
            x = self.dropout(x)

        # output layer
        x = self.fc4(x)
        return x
