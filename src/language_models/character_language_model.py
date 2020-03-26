import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from language_models.dataset.dinosaurs import WordDataset, CharacterVocabulary
from torch.utils.data import random_split, DataLoader

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from argparse import Namespace


class CharacterLanguageModel(pl.LightningModule):
    def __init__(self, hparams):
        super(CharacterLanguageModel, self).__init__()

        self.hparams = hparams
        vocab_size = hparams.vocab_size

        self.embed = nn.Embedding.from_pretrained(torch.eye(vocab_size, vocab_size))
        self.rnn = nn.GRU(input_size=vocab_size, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        e = self.embed(x)
        out, _ = self.rnn(e)
        y = self.fc(out)

        return y

    def training_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)
        output = output.transpose(1, 2)
        loss = self.criterion(output, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output = self.forward(x)
        output = output.transpose(1, 2)
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
        return super().on_save_checkpoint(checkpoint)


if __name__ == '__main__':
    vocab = CharacterVocabulary()
    dset = WordDataset(vocab=vocab)

    train_size = int(0.8 * len(dset))
    train_dset, val_dset = random_split(dset, [train_size, len(dset) - train_size])

    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False)

    n = Namespace()
    n.vocab_size = vocab.get_vocab_size()

    model = CharacterLanguageModel(n)

    trainer = pl.Trainer(
        gradient_clip=1.0,
        accumulate_grad_batches=16,
        overfit_pct=0.1,
        early_stop_callback=EarlyStopping(
            monitor='val_loss', strict=True, verbose=True
        ),
        checkpoint_callback=ModelCheckpoint('models/{epoch}', verbose=True),
    )

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
