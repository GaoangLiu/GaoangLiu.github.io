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

lens = []
for _, text in train_dataset:
    lens.append(len(tokenizer(text)))
print('max length', max(lens), 'min length', min(lens), 'average length',
      sum(lens) / len(lens))

vocab = build_vocab_from_iterator(build_vocabulary(
    [train_dataset, test_dataset]),
                                  min_freq=1,
                                  specials=["<UNK>"])

vocab.set_default_index(vocab["<UNK>"])
train_dataset, test_dataset = to_map_style_dataset(
    train_dataset), to_map_style_dataset(test_dataset)

target_classes = ["World", "Sports", "Business", "Sci/Tech"]


def vectorize_batch(batch, max_length):
    y, x = list(zip(*batch))
    x = [vocab(tokenizer(text)) for text in x]
    x = [
        tokens + ([0] * (max_length - len(tokens)))
        if len(tokens) < max_length else tokens[:max_length] for tokens in x
    ]

    # We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]
    return torch.tensor(x, dtype=torch.int32), torch.tensor(y) - 1


vocab_size = len(vocab)


def get_dataloaders(max_length: int, batch_size: int):
    collate_fn = lambda batch: vectorize_batch(batch, max_length)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=collate_fn,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn)
    return train_loader, test_loader
