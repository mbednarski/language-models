import torch
from torch.utils.data import Dataset
from pathlib import Path


class CharacterVocabulary:
    SOS_TOKEN = '^'
    EOS_TOKEN = '$'

    def __init__(self):
        self.idx2char = list(
            self.SOS_TOKEN + self.EOS_TOKEN + 'abcdefghijklmnopqrstuvwxyz_'
        )
        self.char2idx = {c: i for i, c in enumerate(self.idx2char)}

    def __len__(self):
        return len(self.idx2char)

    def get_vocab_size(self):
        return len(self.char2idx)

    def encode(self, s: str, include_special_tokens=False) -> torch.LongTensor:
        if include_special_tokens:
            s = self.SOS_TOKEN + s + self.EOS_TOKEN
        values = [self.char2idx[c] for c in s]
        return torch.LongTensor(values)

    def decode(self, t: torch.LongTensor) -> str:
        decoded = [self.idx2char[i] for i in t]
        return ''.join(decoded)


class WordDataset(Dataset):
    def __init__(self, vocab=None, dset_path: Path = None):
        if vocab is None:
            self.vocab = CharacterVocabulary()
        else:
            self.vocab = vocab

        if dset_path is None:
            dset_path = Path('data/raw/dinosaurs.txt')
        else:
            dset_path = dset_path

        lines = []
        with dset_path.open('rt') as f:
            lines = [l.strip() for l in f.readlines()]
        self.lines = [l for l in lines if len(l) > 0]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        x = self.lines[idx]
        y = self.lines[idx]

        x = self.vocab.encode(x, include_special_tokens=True)[:-1]
        y = self.vocab.encode(y, include_special_tokens=True)[1:]

        return x, y
