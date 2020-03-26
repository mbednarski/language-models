import torch
from torch.utils.data import Dataset

class CharacterVocabulary:
  SOS_TOKEN = '^'
  EOS_TOKEN = '$'

  def __init__(self):
    self.idx2char = list(self.SOS_TOKEN + self.EOS_TOKEN + 'abcdefghijklmnopqrstuvwxyz')
    self.char2idx = {c:i for i, c in enumerate(self.idx2char)}

  def __len__(self):
    return len(self.idx2char)

  def encode(self, s:str, include_special_tokens=False) -> torch.LongTensor:
    if include_special_tokens:
      s = self.SOS_TOKEN + s + self.EOS_TOKEN
    values = [self.char2idx[c] for c in s]
    return torch.LongTensor(values)
  
  def decode(self, t:torch.LongTensor) -> str:
    decoded = [self.idx2char[i] for i in t]
    return ''.join(decoded)

class DinoDataset(Dataset):
  def __init__(self):
    lines = []
    with open('data/raw/dinosaurs.txt', 'rt') as f:
      lines = f.readlines()
    self.lines = [SOS_TOKEN +  l.lower().strip() + EOS_TOKEN  for l in lines]

    self.idx2char = list(SOS_TOKEN + EOS_TOKEN + 'abcdefghijklmnopqrstuvwxyz')
    self.char2idx = {c:i for i, c in enumerate(self.idx2char)}

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    # TODO: encode to tensor only once and return two slices of it as x and y
    x = self.lines[idx][:-1]
    y = self.lines[idx][1:]

    x = torch.LongTensor([self.char2idx[xx] for xx in x])
    y = torch.LongTensor([self.char2idx[yy] for yy in y])

    return x, y

  def get_vocab_size(self):
      return len(self.char2idx)

  def decode(self, encoded:torch.Tensor) -> str:
    decoded = ''.join([self.idx2char[idx] for idx in encoded])
    return decoded

  def encode(self, s:str) -> str:
    encoded = torch.LongTensor([self.char2idx[xx] for xx in s])

    return encoded