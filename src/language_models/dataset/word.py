import torch
from torch.utils.data import Dataset
from pathlib import Path
from language_models.vocabulary import CharacterVocabulary


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
