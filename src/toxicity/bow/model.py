import numpy as np
import polars as pl

import torch
import torch.nn as nn
from torch.utils.data import Dataset

OTHER_TOKEN = '[OTHER]'

def encode_tokens(tokens: list[str], vocab_index: dict[str, int]):
    zs = np.zeros(len(vocab_index))
    for t in tokens:
        if t in vocab_index:
            zs[vocab_index[t]] += 1
        else:
            zs[vocab_index[OTHER_TOKEN]] = +1
    return zs

class BoWModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super(BoWModel,self).__init__()
        self.fc1 = nn.Linear(vocab_size, 512,dtype=torch.float32)
        self.fc2 = nn.Linear(512,512,dtype=torch.float32)
        self.fc3 = nn.Linear(512,256,dtype=torch.float32)
        self.fc4 = nn.Linear(256,1,dtype=torch.float32)

    def forward(self, data):
        temp = torch.relu(self.fc1(data))
        temp = torch.relu(self.fc2(temp))
        temp = torch.relu(self.fc3(temp))
        return self.fc4(temp)


class BoWDataset(Dataset):
    def __init__(self, df: pl.DataFrame, token_column: str, 
                 target_column: str, vocab_index: dict[str, int]):
        self.tokens = df[token_column]
        self.target = df[target_column]
        self.vocab_index = vocab_index

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return {
            "data": torch.tensor(
                encode_tokens(self.tokens[index], self.vocab_index), 
                dtype=torch.float32,
            ),
            "target": torch.tensor(self.target[index], dtype=torch.float32),
        }