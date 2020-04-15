import torch
from torch.utils.data import Dataset
from pathlib import Path
from language_models.tokenizer import FixedTokenizer


class Seq2SeqDataset(Dataset):
    def __init__(self, source_tokenizer, sources, target_tokenizer, targets):
        self.x = sources
        self.y = targets
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        x = self.source_tokenizer.encode(x, include_special_tokens=True)
        y = self.target_tokenizer.encode(y, include_special_tokens=True)

        return x, y
