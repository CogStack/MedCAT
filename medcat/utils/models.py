from re import I
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

from medcat.utils.model_utils import BertPooler, BertPreTrainingHeads, BertEmbeddings, BertEncoder

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

class BertModel_RelationExtracation(BertPreTrainedModel):
    def __init__(self, config, model_size, task=None, n_classes_=None):
        super(BertModel_RelationExtracation, self).__init__(config)
        self.config = config
        self.task = task
        self.model_size = model_size
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.model_size = model_size
        self.init_weights()
        
        print("Model config: ", self.config)
        if self.task is None:
            self.activation = nn.Tanh()
            ### LM head ###
            self.cls = BertPreTrainingHeads(config)

        elif self.task == 'classification':
            self.n_classes_ = n_classes_
            if self.model_size == 'bert-base-uncased':
                self.classification_layer = nn.Linear(1536, n_classes_)
            elif self.model_size == 'bert-large-uncased':
                self.classification_layer = nn.Linear(2048, n_classes_)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, \
                Q=None, e1_e2_start=None, pooled_output=None):
        """ Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_extended_attention_mask = encoder_attention_mask
        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        
        ### two heads: LM and blanks ###
        
        blankv1v2 = sequence_output[:, e1_e2_start, :]
        buffer = []

        for i in range(blankv1v2.shape[0]): # iterate batch & collect
            v1v2 = blankv1v2[i, i, :, :] # blankv1v2[i] 
            v1v2 = torch.cat((v1v2[0], v1v2[1]))
            buffer.append(v1v2)
        del blankv1v2
        v1v2 = torch.stack([a for a in buffer], dim=0)
        del buffer
        
        if self.task is None:
            blanks_logits = self.activation(v1v2) # self.blanks_linear(- torch.log(Q)
            lm_logits = self.cls(sequence_output, pooled_output=pooled_output)
            return blanks_logits, lm_logits
        elif self.task == 'classification':
            classification_logits = self.classification_layer(v1v2)
            return classification_logits
        elif self.task == 'fewrel':
            return v1v2
