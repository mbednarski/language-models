import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.utils.rnn as ru
from language_models.dataset.word import WordDataset
from torch.utils.data import random_split, DataLoader

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from argparse import Namespace
from language_models.tokenizer import FixedTokenizer
from pathlib import Path


class CharacterLanguageModel(pl.LightningModule):
    def __init__(self, hparams, padding_token='<pad>'):
        super(CharacterLanguageModel, self).__init__()

        self.hparams = hparams
        vocab_size = hparams.vocab_size

        self.embed = nn.Embedding.from_pretrained(torch.eye(vocab_size, vocab_size))
        self.rnn = nn.GRU(input_size=vocab_size, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)
        self.padding_token = padding_token

        self.criterion = nn.CrossEntropyLoss(
            reduction='mean', ignore_index=padding_token
        )

    def forward(self, x, hidden_state=None):
        x.rename_(None)
        e = self.embed(x)
        out, hidden = self.rnn(e, hidden_state)
        out.rename_('B', 'L', 'H')
        y = self.fc(out)
        y.rename_('B', 'L', 'V')

        return y, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch

        output, _ = self.forward(x)  # BxLxV
        output.rename_(None)
        output = output.transpose(1, 2)
        # output ('B', 'L', 'V')

        y.rename_(None)
        # y: BxL

        loss = self.criterion(output, y)

        logs = {
            'train_loss': loss,
        }
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output, _ = self.forward(x)  # BxLxV
        output.rename_(None)
        output = output.transpose(1, 2)
        # output ('B', 'L', 'V')

        y.rename_(None)
        # y: BxL

        loss = self.criterion(output, y)

        logs = {'val_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}

        return {'loss': avg_loss, 'log': logs}

    def configure_optimizer(self):
        return optim.Adam(self.parameters())

    def on_save_checkpoint(self, checkpoint):
        checkpoint['padding_token'] = self.padding_token
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        self.padding_token = checkpoint['padding_token']


if __name__ == '__main__':
    tokenizer = FixedTokenizer(
        allowed_tokens=list(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZĄąĆćĘęŁłŃńÓóŚśŹźŻż -."'
        )
    )
    dset = WordDataset(tokenizer=tokenizer, dset_path=Path('data/raw/pol_cities.txt'))

    train_size = int(0.8 * len(dset))
    train_dset, val_dset = random_split(dset, [train_size, len(dset) - train_size])

    train_loader = DataLoader(
        train_dset, batch_size=16, shuffle=True, collate_fn=tokenizer.collate_sequences
    )
    val_loader = DataLoader(
        val_dset, batch_size=16, shuffle=False, collate_fn=tokenizer.collate_sequences
    )

    n = Namespace()
    n.vocab_size = tokenizer.get_vocab_size()

    model = CharacterLanguageModel(
        n, padding_token=tokenizer.tok2idx[tokenizer.PAD_TOKEN]
    )

    trainer = pl.Trainer(
        gpus=[0],
        gradient_clip=1.0,
        max_epochs=101,
        early_stop_callback=EarlyStopping(
            monitor='val_loss', strict=True, verbose=True, patience=10
        ),
        checkpoint_callback=ModelCheckpoint('models/{epoch}', verbose=True),
    )

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
