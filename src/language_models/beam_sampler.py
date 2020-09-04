import torch
import pytorch_lightning as pl

from language_models.character_language_model import CharacterLanguageModel
from language_models.tokenizer import FixedTokenizer

tokenizer = FixedTokenizer(
    allowed_tokens=list(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZĄąĆćĘęŁłŃńÓóŚśŹźŻż -."'
    )
)

class BeamSampler:
    def __init__(
        self, model: pl.LightningModule, tokenizer: FixedTokenizer, strategy: str
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.strategy = strategy

    # kompensacja długości
    # list -> heap
    # beam size vs max proba
    # replace * by log

    def beam(self):
        k=10
        seed = ''
        pathes = [(1.0, seed)]
        for step in range(50):
            new_pathes = []
            for p in pathes:
                # generowanie następników
                candidates = self.expand_node(p[1])
                if len(candidates) == 0:
                    new_pathes.append(p)
                else:
                    for i, c in enumerate(candidates):
                        patch_score, patch_list = p
                        new_score = patch_score * c
                        new_patch = patch_list + tokenizer.idx2tok[i]
                        new_pathes.append((new_score, new_patch))
            

            new_pathes = sorted(new_pathes, reverse=True)
            new_pathes = new_pathes[:k]
            pathes = new_pathes

        pass


    def expand_node(self, seed):
        if len(seed)>0 and seed[-1] == self.tokenizer.EOS_TOKEN:
            return []

        seed = list(seed)
        input = self.tokenizer.encode(
            [self.tokenizer.SOS_TOKEN] + seed, include_special_tokens=False
        )
        input = input.unsqueeze(0)

        sampled_sentence = seed

        self.model.eval()
        hidden = None
        with torch.no_grad():           
            output, next_hidden = self.model.forward(input, hidden)

            hidden = next_hidden

            output = output[0, -1, :].squeeze()
            probas = torch.softmax(output, dim=0)

            probas.rename_(None)

            return probas

    def sample_sentence(self, max_len: int = 20, seed: str = '', strategy='greedy'):
        seed = list(seed)
        if strategy not in {'greedy', 'proba'}:
            raise ValueError()
        input = self.tokenizer.encode(
            [self.tokenizer.SOS_TOKEN] + seed, include_special_tokens=False
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

                probas.rename_(None)

                if strategy == 'proba':
                    sampled_idx = torch.multinomial(probas, 1).squeeze()
                elif strategy == 'greedy':
                    sampled_idx = torch.argmax(probas)
                else:
                    assert False

                sampled_char = self.tokenizer.decode(sampled_idx.unsqueeze(0))

                if sampled_char == self.tokenizer.EOS_TOKEN:
                    break

                sampled_sentence += sampled_char
                input = sampled_idx.unsqueeze(0).unsqueeze(0)

        return sampled_sentence

if __name__ == '__main__':

    model = CharacterLanguageModel.load_from_checkpoint('models/epoch=84.ckpt')
    model.eval()
    model.freeze()


    sampler = BeamSampler(model, tokenizer, strategy='proba')
    sampler.beam()


    while True:
        seed = input('Provide seed: ')
        print(''.join(sampler.sample_sentence(strategy='proba', seed=seed)))
