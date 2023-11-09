import gc

import torch
import torchtext
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import test_loader, train_loader,vocab_size


class RNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 target_classes: int,
                 embed_len: int = 50,
                 hidden_dim: int = 50,
                 n_layers: int = 1):
        super(RNNClassifier, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_len)
        self.rnn = nn.RNN(input_size=embed_len,
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          batch_first=True)
        self.linear = nn.Linear(hidden_dim, target_classes)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, X_batch):
        embeddings = self.embedding_layer(X_batch)
        output, hidden = self.rnn(
            embeddings,
            torch.randn(self.n_layers, len(X_batch), self.hidden_dim))
        return self.linear(output[:, -1])

def calculate_metrics(model, loss_fn, val_loader):
    with torch.no_grad():
        Y_shuffled, ypreds, losses = [], [], []
        for X, Y in val_loader:
            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())

            Y_shuffled.append(Y)
            ypreds.append(preds.argmax(dim=-1))

        Y_shuffled = torch.cat(Y_shuffled)
        ypreds = torch.cat(ypreds)

        print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid Acc  : {:.3f}".format(
            accuracy_score(Y_shuffled.detach().numpy(),
                           ypreds.detach().numpy())))


def train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
    for i in range(1, epochs + 1):
        losses = []
        for X, Y in tqdm(train_loader):
            ypreds = model(X)

            loss = loss_fn(ypreds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        calculate_metrics(model, loss_fn, val_loader)


epochs = 15
learning_rate = 3e-3

loss_fn = nn.CrossEntropyLoss()
rnn_classifier = RNNClassifier(vocab_size=vocab_size, target_classes=4)
optimizer = Adam(rnn_classifier.parameters(), lr=learning_rate)

train_model(rnn_classifier, loss_fn, optimizer, train_loader, test_loader,
           epochs)
