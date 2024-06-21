import numpy as np
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import Dataset


def encode_tokens(
    tokens: list[str],
    embeddings: dict[str, list[int]],
    emb_dim: int,
    seq_len: int,
) -> np.array:
    hells_v = np.zeros((seq_len, emb_dim), dtype=np.float32)

    for i, token in enumerate(tokens):
        if i >= seq_len:
            break

        if token in embeddings:
            hells_v[i] = embeddings[token]
        else:
            hells_v[i] = np.zeros(emb_dim)

    return hells_v


class EmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        seq_len: int,
        hidden_size: int = 512,
    ):
        super(EmbeddingModel, self).__init__()
        self.rnn = nn.RNN(emb_dim, hidden_size, nonlinearity="relu", batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1, dtype=torch.float32)

    def forward(self, data):
        out, h_n = self.rnn(data)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        return out


class EmbeddingModelLSTM(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        seq_len: int,
        hidden_size: int = 256,
        num_layers: int = 1,
    ):
        super(EmbeddingModelLSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            emb_dim, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 1, dtype=torch.float32, bias=True)

    def forward(self, data):
        hidden = torch.randn(self.num_layers, data.size(0), self.hidden_size).to(
            data.device
        )
        carry = torch.randn(self.num_layers, data.size(0), self.hidden_size).to(
            data.device
        )

        out, (hidden, carry) = self.lstm(data, (hidden, carry))
        out = out[:, -1]
        out = torch.relu(self.fc1(out))
        return out


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        token_column: str,
        target_column: str,
        embeddings: dict[str, list[int]],
        emb_dim: int,
        seq_len: int,
    ):
        self.tokens = df[token_column]
        self.target = df[target_column]
        self.embeddings = embeddings
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.cache = {}

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        self.cache[index] = {
            "data": torch.tensor(
                encode_tokens(
                    self.tokens[index],
                    embeddings=self.embeddings,
                    emb_dim=self.emb_dim,
                    seq_len=self.seq_len,
                ),
                dtype=torch.float32,
            ),
            "target": torch.tensor(self.target[index], dtype=torch.float32),
        }

        return self.cache[index]
