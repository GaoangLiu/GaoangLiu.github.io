#!/usr/bin/env python3
import torch
import torchtext
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")


def build_vocabulary(datasets):
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)


train_dataset, test_dataset = torchtext.datasets.AG_NEWS()
vocab = build_vocab_from_iterator(build_vocabulary(
    [train_dataset, test_dataset]),
                                  min_freq=1,
                                  specials=["<UNK>"])

vocab.set_default_index(vocab["<UNK>"])
train_dataset, test_dataset = to_map_style_dataset(
    train_dataset), to_map_style_dataset(test_dataset)

target_classes = ["World", "Sports", "Business", "Sci/Tech"]


def vectorize_batch(batch, max_length):
    Y, X = list(zip(*batch))
    X = [vocab(tokenizer(text)) for text in X]
    X = [
        tokens + ([0] * (max_length - len(tokens)))
        if len(tokens) < max_length else tokens[:max_length] for tokens in X
    ]  ## Bringing all samples to max_length length.

    return torch.tensor(X, dtype=torch.int32), torch.tensor(
        Y
    ) - 1  ## We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]

vocab_size = len(vocab)
def get_dataloaders(max_length:int):
    collate_fn = lambda batch: vectorize_batch(batch, max_length)
    train_loader = DataLoader(train_dataset,
                            batch_size=1024,
                            collate_fn=collate_fn,
                            shuffle=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=1024,
                            collate_fn=collate_fn)
    return train_loader, test_loader



