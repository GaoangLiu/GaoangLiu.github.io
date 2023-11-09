import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer


# Define the RNN Classifier model
class RNNClassifier(pl.LightningModule):

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['label']
        output = self(input_ids)
        loss = self.criterion(output, labels)
        return {'loss': loss}

    def configure_optimizers(self):
        return optim.Adam(self.parameters())


class IMDbDataset(Dataset):

    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        tokenized = self.tokenizer(text,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_seq_length,
                                   return_tensors='pt')
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'label': self.data[idx]['label']
        }


imdb_dataset = load_dataset(
    '/home/gaoang/.cache/huggingface/datasets/ttxy___emotion/')
train_dataset = imdb_dataset['validation']
valid_dataset = imdb_dataset['test']
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_seq_length = 128

train_dataset = IMDbDataset(train_dataset, tokenizer, max_seq_length)
valid_dataset = IMDbDataset(valid_dataset, tokenizer, max_seq_length)

# Initialize the RNNClassifier model
input_size = tokenizer.vocab_size
hidden_size = 128
output_size = 5
num_layers = 1
rnn_classifier = RNNClassifier(input_size, hidden_size, output_size, num_layers)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# Train the model
trainer = pl.Trainer(max_epochs=5)
trainer.fit(rnn_classifier, train_loader, valid_loader)
