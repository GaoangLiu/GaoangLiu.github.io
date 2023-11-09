import torch
from torch import nn
from torch.nn import functional as F


class SingleRNN(nn.Module):
    """ One layer RNN classifier.
    """

    def __init__(self,
                 vocab_size: int,
                 target_classes: int,
                 embed_len: int = 50,
                 hidden_dim: int = 50,
                 bidirectional: bool = False,
                 num_layers: int = 1):
        super(SingleRNN, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(hidden_dim * 2, target_classes)
        else:
            self.linear = nn.Linear(hidden_dim, target_classes)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

    def forward(self, x_batch):
        embeddings = self.embedding_layer(x_batch)
        if self.bidirectional:
            init_hidden = torch.randn(self.num_layers * 2, len(x_batch),
                                        self.hidden_dim)
        else:
            init_hidden = torch.randn(self.num_layers, len(x_batch),
                                        self.hidden_dim)
        output, hidden = self.rnn(embeddings, init_hidden)
        return self.linear(output[:, -1])
