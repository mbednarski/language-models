import torch
from torch.nn.utils.rnn import pad_sequence


class FixedTokenizer:
    SOS_TOKEN = '!'
    EOS_TOKEN = '@'
    PAD_TOKEN = '#'

    DEFAULT_ALLOWED_TOKENS = list('abcdefghijklmnopqrstuvwxyz_')

    def __init__(self, allowed_tokens=None):
        if allowed_tokens is None:
            self.allowed_tokens = self.DEFAULT_ALLOWED_TOKENS
        else:
            self.allowed_tokens = allowed_tokens

        self.idx2tok = list(
            [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN] + self.allowed_tokens
        )
        self.tok2idx = {t: i for i, t in enumerate(self.idx2tok)}

    def __len__(self):
        return len(self.idx2tok)

    def get_vocab_size(self):
        return len(self.tok2idx)

    def encode(self, s: str, include_special_tokens=False) -> torch.LongTensor:
        if include_special_tokens:
            s = [self.SOS_TOKEN] + list(s) + [self.EOS_TOKEN]
        values = [self.tok2idx[c] for c in s]
        return torch.LongTensor(values)

    def decode(self, t: torch.LongTensor) -> str:
        decoded = [self.idx2tok[i] for i in t]
        return ''.join(decoded)

    def collate_sequences(self, batch):
        x, y = zip(*batch)

        x = list(x)
        y = list(y)

        x = pad_sequence(
            x, batch_first=True, padding_value=self.tok2idx[self.PAD_TOKEN]
        )
        y = pad_sequence(
            y, batch_first=True, padding_value=self.tok2idx[self.PAD_TOKEN]
        )

        x.rename_('B', 'L')
        y.rename_('B', 'L')

        return x, y
