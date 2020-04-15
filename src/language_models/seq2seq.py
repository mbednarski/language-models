# %%
from itertools import chain
from language_models.tokenizer import FixedTokenizer
from language_models.dataset.seq2seq import Seq2SeqDataset
from nltk import wordpunct_tokenize, FreqDist
import pandas as pd
from collections import Counter
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.rnn as ru
from torch.optim import Adam
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self, input_size:int):
        super(Encoder, self).__init__()

        self.rnn = nn.GRU(input_size, 64)
    
    def forward(self, x):
        out, hidden = self.rnn(x)
        return hidden

class Decoder(nn.Module):
    def __init__(self, out_features):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(64, out_features)
    
    def forward(self, x):
        return self.fc(x)

class Seq2SeqModel(pl.LightningModule):
    def __init__(self, input_vocab_size:int, output_vocab_size:int):
        super(Seq2SeqModel, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.eye(input_vocab_size), padding_idx=0)
        self.encoder = Encoder(input_vocab_size)
        self.decoder = Decoder(output_vocab_size)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: BxL

        e = self.embed(x)
        # e: BxLxVin

        context = self.encoder(e)

        return context

    def training_step(self, batch, batch_idx):
        x, y = batch

        c = self.forward(x)

        loss = self.criterion(y, c)

        return{
            "loss": loss
        }

    @pl.data_loader
    def train_dataloader(self):
        df = pd.read_csv('data/raw/equations.csv', sep=';')
        df['textual_tokenized'] = df['textual'].apply(wordpunct_tokenize)
        textual_counter = Counter(chain(*df['textual_tokenized'].tolist()))
        textual_tokens = list(sorted(textual_counter.keys()))
        textual_tokenizer = FixedTokenizer(allowed_tokens=textual_tokens)

        df['numerical_tokenized'] = df['numerical'].apply(lambda x: list(x))
        numerical_counter = Counter(chain(*df['numerical_tokenized'].tolist()))
        numerical_tokens = list(sorted(numerical_counter.keys()))

        numerical_tokenizer = FixedTokenizer(allowed_tokens=numerical_tokens)

        dset = Seq2SeqDataset(numerical_tokenizer, df['numerical_tokenized'].tolist(),
        textual_tokenizer, df['textual_tokenized'].tolist()
        )

        return DataLoader(dset, batch_size=1, shuffle=False)

if __name__ == '__main__':
    model = Seq2SeqModel(17, 38)

    trainer = pl.Trainer(gpus=None)
    trainer.fit(model)

    print(dset[32])

