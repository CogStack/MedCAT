import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self, embeddings, padding_idx):
        super(LSTM, self).__init__()
        self.padding_idx = padding_idx
        # Get the required sizes
        vocab_size = len(embeddings)
        embedding_size = len(embeddings[0])

        # Initialize embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.embeddings.load_state_dict({'weight': embeddings})
        # Disable training for the embeddings - IMPORTANT
        self.embeddings.weight.requires_grad = False

        hidden_size = 300
        bid = True # Is the network bidirectional

        # Create the RNN cell - devide 
        self.rnn = nn.LSTM(input_size=300,
                           hidden_size=hidden_size // (2 if bid else 1),
                           num_layers=2,
                           dropout=0.5,
                           bidirectional=bid)
        self.fc1 = nn.Linear(hidden_size, 2)

        self.d1 = nn.Dropout(0.5)


    def forward(self, x, cpos):
        # Get the mask from x
        mask = x != self.padding_idx

        # Embed the input: from id -> vec
        x = self.embeddings(x) # x.shape = batch_size x sequence_length x emb_size

        # Tell RNN to ignore padding and set the batch_first to True
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int().view(-1), batch_first=True, enforce_sorted=False)

        # Run 'x' through the RNN
        x, hidden = self.rnn(x)

        # Add the padding again
        x, hidden = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Get what we need
        row_indices = torch.arange(0, x.size(0)).long()
        x = x[row_indices, cpos, :]

        # Push x through the fc network and add dropout
        x = self.d1(x)
        #x = self.fc1(x)
        x = self.fc1(x)

        return x
