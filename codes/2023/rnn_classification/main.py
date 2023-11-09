import gc
import itertools

import codefast as cf
import torch
from rich import print
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm

from dataset import get_dataloaders, vocab_size
from model import SingleRNN


def validate_model(model, loss_fn, val_loader):
    with torch.no_grad():
        y_shuffled, ypreds, losses = [], [], []
        for x, y in val_loader:
            preds = model(x)
            loss = loss_fn(preds, y)
            losses.append(loss.item())

            y_shuffled.append(y)
            ypreds.append(preds.argmax(dim=-1))

        y_shuffled = torch.cat(y_shuffled)
        ypreds = torch.cat(ypreds)

        print("Valid loss : {:.3f}".format(torch.tensor(losses).mean()))
        print("Valid acc  : {:.3f}".format(
            accuracy_score(y_shuffled.detach().numpy(),
                           ypreds.detach().numpy())))


def train_model(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
    for i in range(1, epochs + 1):
        print('Epoch {}/{}'.format(i, epochs))
        losses = []
        for x, y in tqdm(train_loader):
            ypreds = model(x)

            loss = loss_fn(ypreds, y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
        validate_model(model, loss_fn, val_loader)
    return model


def test_model(model, test_loader):
    with torch.no_grad():
        ypreds = []
        ys = []
        for x, y in test_loader:
            preds = model(x)
            ypreds.append(preds.argmax(dim=-1))
            ys.append(y)
        ypreds = torch.cat(ypreds)
        ys = torch.cat(ys)
    reports = classification_report(ys, ypreds, output_dict=True)
    return reports


def _experiment_once(max_length: int, num_layers: int, bidirectional: bool):
    epochs = 15
    learning_rate = 1e-3
    train_loader, test_loader = get_dataloaders(max_length)

    loss_fn = nn.CrossEntropyLoss()
    performace = []
    for _ in range(3):     # repeat 3 times to get average performance
        model = SingleRNN(vocab_size=vocab_size,
                          target_classes=4,
                          embed_len=max_length,
                          bidirectional=bidirectional,
                          num_layers=num_layers)
        print(model)
        optimizer = Adam(model.parameters(), lr=learning_rate)

        model = train_model(model, loss_fn, optimizer, train_loader,
                            test_loader, epochs)

        reports = test_model(model, test_loader)
        print(reports['weighted avg'])
        del model
        gc.collect()
        performace.append(reports['weighted avg'])
    return {
        'precision':
        sum([x['precision'] for x in performace]) / len(performace),
        'recall': sum([x['recall'] for x in performace]) / len(performace),
        'f1-score': sum([x['f1-score'] for x in performace]) / len(performace),
    }


def run_experiments():
    max_length = [25, 50, 100]
    num_layers = [1, 2, 3]
    bidirectional = [True, False]

    results = []
    for mlen, nlayer, bid in itertools.product(max_length, num_layers,
                                               bidirectional):
        performace = _experiment_once(mlen, nlayer, bid)
        performace['max_length'] = mlen
        performace['num_layers'] = nlayer
        performace['bidirectional'] = bid
        results.append(performace)
    cf.js.write(results, 'experiment_results.json')


if __name__ == '__main__':
    run_experiments()
