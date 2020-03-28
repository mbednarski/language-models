import torch


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
