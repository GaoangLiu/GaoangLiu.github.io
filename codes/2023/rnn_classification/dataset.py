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

max_words = 25

def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = [vocab(tokenizer(text)) for text in X]
    X = [
        tokens + ([0] * (max_words - len(tokens)))
        if len(tokens) < max_words else tokens[:max_words] for tokens in X
    ]  ## Bringing all samples to max_words length.

    return torch.tensor(X, dtype=torch.int32), torch.tensor(
        Y
    ) - 1  ## We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]


train_loader = DataLoader(train_dataset,
                          batch_size=1024,
                          collate_fn=vectorize_batch,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=1024,
                         collate_fn=vectorize_batch)
vocab_size = len(vocab)
