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

    return hells_v.flatten()


class EmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        seq_len: int,
    ):
        super(EmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(emb_dim * seq_len, 512, dtype=torch.float32)
        self.fc2 = nn.Linear(512, 512, dtype=torch.float32)
        self.fc3 = nn.Linear(512, 256, dtype=torch.float32)
        self.fc4 = nn.Linear(256, 1, dtype=torch.float32)

    def forward(self, data):
        temp = torch.relu(self.fc1(data))
        temp = torch.relu(self.fc2(temp))
        temp = torch.relu(self.fc3(temp))
        return self.fc4(temp)


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
