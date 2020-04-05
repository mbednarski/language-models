import torch
import pytorch_lightning as pl

from language_models.character_language_model import CharacterLanguageModel
from language_models.vocabulary import CharacterVocabulary


class LanguageModelSampler:
    def __init__(
        self, model: pl.LightningModule, vocab: CharacterVocabulary, strategy: str
    ):
        self.model = model
        self.vocab = vocab

        self.strategy = strategy

    def sample_sentence(self, max_len: int = 20, seed: str = '', strategy='greedy'):

        if strategy not in {'greedy'}:
            raise ValueError()
        input = self.vocab.encode(
            self.vocab.SOS_TOKEN + seed, include_special_tokens=False
        )
        input = input.unsqueeze(0)

        sampled_sentence = seed

        self.model.eval()
        hidden = None
        with torch.no_grad():
            for _ in range(max_len):
                output, next_hidden = self.model.forward(input, hidden)

                hidden = next_hidden

                output = output[0, -1, :].squeeze()
                probas = torch.softmax(output, dim=0)

                # sampled_idx = torch.multinomial(probas, 1).squeeze()

                sampled_idx = torch.argmax(probas)

                sampled_char = self.vocab.decode(sampled_idx.unsqueeze(0))

                if sampled_char == self.vocab.EOS_TOKEN:
                    break

                sampled_sentence += sampled_char
                input = sampled_idx.unsqueeze(0).unsqueeze(0)

        return sampled_sentence
