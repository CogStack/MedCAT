from re import I
import torch
from torch import nn

from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler, BertPreTrainingHeads, BertEncoder, BertModel
from transformers.models.bert.configuration_bert import BertConfig

class BertModel_RelationExtraction(BertPreTrainedModel):
    def __init__(self, config : BertConfig, model_size, task : str = "train", n_classes=None):
        super(BertModel_RelationExtraction, self).__init__(config)

        self.config = config
        print("Model config: ", self.config)

        self.task = task
        self.model_size = model_size
        self.embeddings = BertEmbeddings(config)
        #self.encoder = BertEncoder(config)
        self.model= BertModel(config)
        self.pooler = BertPooler(config)
        self.drop_out = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()
        
        if self.task is "pretrain":
            self.activation = nn.Tanh()
            ### LM head ###
            self.cls = BertPreTrainingHeads(config)

        elif self.task == "train":
            self.n_classes = n_classes
            self.classification_layer = nn.Linear(config.hidden_size, n_classes)
            
            if self.model_size == 'bert-base-uncased':
                self.classification_layer = nn.Linear(1536, self.n_classes)
            elif self.model_size == 'bert-large-uncased':
                self.classification_layer = nn.Linear(2048, self.n_classes)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

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
        
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                  encoder_hidden_states=encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask)
         
        sequence_output = model_output[0] # (batch_size, #no of features, sequence_length, hidden_size)
        pooled_output = model_output[1]

        e1e2_output = sequence_output[:, e1_e2_start, :] 
        
        buffer = []
        for sample_index in range(e1e2_output.shape[0]):
            e1e2 = e1e2_output[sample_index, sample_index, :, :]
            e1e2 = torch.cat((e1e2[0], e1e2[1]))
            buffer.append(e1e2)

        e1e2_logits = torch.stack([a for a in buffer], dim=0)

        classification_logits = None

        if self.task == "train":
            classification_logits = self.classification_layer(e1e2_logits)

        return classification_logits