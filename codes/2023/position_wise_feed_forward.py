import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, inner_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_dim, inner_dim), nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Linear(inner_dim, hidden_dim))

    def forward(self, x):
        return self.fc(x)


d_model = 512
d_ff = 2048
x = torch.rand(64, 10, d_model)
ff = PositionWiseFeedForward(d_model, d_ff)
y = ff(x)
print(y.shape)
# torch.Size([64, 10, 512])
