import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

class LSTM(nn.Module):
    def __init__(self, embeddings, padding_idx, nclasses=2, bid=True, input_size=300,
                 num_layers=2, hidden_size=300, dropout=0.5):
        super(LSTM, self).__init__()
        self.padding_idx = padding_idx
        # Get the required sizes
        vocab_size = len(embeddings)
        embedding_size = len(embeddings[0])

        self.num_layers = num_layers
        self.bid = bid
        self.input_size = input_size
        self.nclasses = nclasses
        self.num_directions = (2 if self.bid else 1)
        self.dropout = dropout

        # Initialize embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.embeddings.load_state_dict({'weight': embeddings})
        # Disable training for the embeddings - IMPORTANT
        self.embeddings.weight.requires_grad = False

        self.hidden_size = hidden_size
        bid = True # Is the network bidirectional

        # Create the RNN cell - devide 
        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size // self.num_directions,
                           num_layers=self.num_layers,
                           dropout=dropout,
                           bidirectional=self.bid)
        self.fc1 = nn.Linear(self.hidden_size, nclasses)

        self.d1 = nn.Dropout(dropout)


    def forward(self, input_ids, center_positions, attention_mask=None, ignore_cpos=False):
        x = input_ids
        # Get the mask from x
        if attention_mask is None:
            mask = x != self.padding_idx
        else:
            mask = attention_mask

        # Embed the input: from id -> vec
        x = self.embeddings(x) # x.shape = batch_size x sequence_length x emb_size

        # Tell RNN to ignore padding and set the batch_first to True
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int().view(-1).cpu(), batch_first=True, enforce_sorted=False)

        # Run 'x' through the RNN
        x, hidden = self.rnn(x)

        # Add the padding again
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Get what we need
        row_indices = torch.arange(0, x.size(0)).long()

        # If this is  True we will always take the last state and not CPOS
        if ignore_cpos:
            x = hidden[0]
            x = x.view(self.num_layers, self.num_directions, -1, self.hidden_size//self.num_directions)
            x = x[-1, :, :, :].permute(1, 2, 0).reshape(-1, self.hidden_size)
        else:
            x = x[row_indices, center_positions, :]

        # Push x through the fc network and add dropout
        x = self.d1(x)
        x = self.fc1(x)

        return x


class BertForMetaAnnotation(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        center_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # (batch_size, sequence_length, hidden_size)

        row_indices = torch.arange(0, sequence_output.size(0)).long()
        sequence_output = sequence_output[row_indices, center_positions, :]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
