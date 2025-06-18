import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

class SentimentAnalyzer(nn.Module):
    def __init__(self, hidden_dim=200, model_name="distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(
            input_size=self.encoder.config.hidden_size,
            hidden_size=hidden_dim,
            batch_first=True,
            nonlinearity='relu'
        )
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Freeze encoder (optional, remove if fine-tuning)
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # Shape: (batch, seq_len, hidden_size)

        rnn_out, _ = self.rnn(embeddings)  # (batch, seq_len, hidden_dim)
        final_hidden = rnn_out[:, -1, :]   # (batch, hidden_dim)
        logits = self.linear(final_hidden)
        probs = self.sigmoid(logits).squeeze(1)  # (batch,)
        return probs


class SentimentDataset(Dataset):
    def __init__(self, tokenizer_name="distilbert-base-uncased", train=True, length=10000, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = load_dataset("stanfordnlp/imdb", split="train" if train else "test")
        self.length = min(length, len(self.dataset))
        self.max_length = max_length
        print(f"Loaded {self.length} samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0), # tokenized text
            "attention_mask": encoding["attention_mask"].squeeze(0), # 1/0, specifies padding vs non-padding
            "label": torch.tensor(item["label"], dtype=torch.float)
        }
    def tokenize_string(self, sentence):
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # tokenized text
            "attention_mask": encoding["attention_mask"].squeeze(0),  # 1/0, specifies padding vs non-padding
        }


def collate_batch(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    return input_ids, attention_mask, labels
